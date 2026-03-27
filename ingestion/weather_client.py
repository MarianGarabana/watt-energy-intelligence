"""
Open-Meteo Weather API Client
==============================
Fetches historical and forecast hourly weather data.

✓ No API key required — completely free.
Docs: https://open-meteo.com/en/docs

We pull the weather features most predictive for energy demand + renewable generation:
  - temperature_2m          → heating/cooling demand driver
  - shortwave_radiation      → solar panel output proxy
  - wind_speed_10m          → wind turbine output proxy
  - cloud_cover              → solar generation dampener
  - precipitation            → hydro generation proxy
  - apparent_temperature     → "feels like" — stronger demand signal than raw temp
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 30

# Grid region → representative (lat, lon) coordinates
REGION_COORDS = {
    "CAL":  (36.7783,  -119.4179),   # California centroid
    "TEX":  (31.9686,   -99.9018),   # Texas centroid
    "NY":   (42.9538,   -75.5268),   # New York state centroid
    "NE":   (42.3601,   -71.0589),   # Boston (New England hub)
    "MIDW": (41.8781,   -87.6298),   # Chicago (Midwest hub)
    "NW":   (47.6062,  -122.3321),   # Seattle (Northwest hub)
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
]


class WeatherClient:
    """
    Client for Open-Meteo API.
    Automatically uses the archive endpoint for historical data (>7 days ago)
    and the forecast endpoint for recent + future data.

    Usage:
        client = WeatherClient()
        df = client.get_weather(region="CAL", days_back=30)
    """

    def __init__(self):
        self.session = requests.Session()

    # ── private ──────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict) -> dict:
        """Make a GET request with retry logic."""
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def _parse_response(self, data: dict, region: str) -> pd.DataFrame:
        """Convert Open-Meteo JSON response into a tidy DataFrame."""
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["timestamp"] = pd.to_datetime(df["time"], utc=True)
        df["region"] = region
        df = df.drop(columns=["time"])

        # reorder: timestamp + region first
        cols = ["timestamp", "region"] + [c for c in df.columns if c not in ("timestamp", "region")]
        return df[cols]

    # ── public ───────────────────────────────────────────────────────────────

    def get_weather(
        self,
        region: str = "CAL",
        days_back: int = 30,
        days_forward: int = 2,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly weather for a grid region (or custom lat/lon).

        Args:
            region:       EIA region code — used to look up coordinates
            days_back:    Days of historical data to fetch
            days_forward: Days of forecast data to include (for feature generation)
            lat/lon:      Override coordinates (optional)

        Returns:
            DataFrame with hourly weather features
        """
        if lat is None or lon is None:
            if region not in REGION_COORDS:
                raise ValueError(
                    f"Unknown region '{region}'. "
                    f"Available: {list(REGION_COORDS.keys())} "
                    f"or pass lat/lon directly."
                )
            lat, lon = REGION_COORDS[region]

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end_date   = (now + timedelta(days=days_forward)).strftime("%Y-%m-%d")

        # Use archive for older data (more complete), forecast for recent
        cutoff_days = 7
        dfs = []

        if days_back > cutoff_days:
            archive_end = (now - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
            logger.info(f"Fetching archive weather for {region}: {start_date} → {archive_end}")
            archive_params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": archive_end,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": "UTC",
            }
            archive_data = self._get(ARCHIVE_URL, archive_params)
            dfs.append(self._parse_response(archive_data, region))

        # Fetch recent + forecast from forecast endpoint
        recent_start = (now - timedelta(days=min(days_back, cutoff_days))).strftime("%Y-%m-%d")
        logger.info(f"Fetching forecast weather for {region}: {recent_start} → {end_date}")
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": recent_start,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "forecast_days": days_forward + 1,
        }
        forecast_data = self._get(FORECAST_URL, forecast_params)
        dfs.append(self._parse_response(forecast_data, region))

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        if df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        df = df.drop_duplicates(subset=["timestamp", "region"]).sort_values("timestamp")
        df = df.reset_index(drop=True)

        logger.info(f"✓ Weather for {region}: {len(df)} hourly records ({start_date} → {end_date})")
        return df

    def get_all_regions(
        self,
        regions: list[str] = list(REGION_COORDS.keys()),
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Fetch weather for all grid regions and concatenate."""
        dfs = []
        for region in regions:
            try:
                dfs.append(self.get_weather(region=region, days_back=days_back))
            except Exception as e:
                logger.error(f"Weather fetch failed for {region}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✓ Total weather records: {len(df)} across {len(regions)} regions")
        return df


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    client = WeatherClient()

    print("\n── Weather (California, last 7 days + 2 day forecast) ──")
    df = client.get_weather(region="CAL", days_back=7, days_forward=2)
    print(df[["timestamp", "temperature_2m", "shortwave_radiation", "wind_speed_10m"]].head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
