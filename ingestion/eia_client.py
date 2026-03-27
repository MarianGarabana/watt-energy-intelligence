"""
EIA (U.S. Energy Information Administration) API Client
========================================================
Fetches hourly electricity demand and generation mix data
from the EIA Open Data API v2.

Free API key: https://www.eia.gov/opendata/

Regions (RTOs) available:
  CAL  - California
  CAR  - Carolinas
  CENT - Central
  FLA  - Florida
  MIDA - Mid-Atlantic
  MIDW - Midwest
  NE   - New England
  NW   - Northwest
  NY   - New York
  SE   - Southeast
  SW   - Southwest
  TEN  - Tennessee
  TEX  - Texas
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"
DEFAULT_REGIONS = ["CAL", "TEX", "NY", "NE", "MIDW", "NW"]
REQUEST_TIMEOUT = 30
RATE_LIMIT_SLEEP = 0.5  # seconds between requests


class EIAClient:
    """
    Client for the EIA Open Data API v2.

    Usage:
        client = EIAClient()
        df = client.get_hourly_demand(region="CAL", days_back=7)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key not found. Set EIA_API_KEY in your .env file.\n"
                "Get a free key at: https://www.eia.gov/opendata/"
            )
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}  # attached to every request

    # ── private ──────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a GET request with retry logic."""
        url = f"{EIA_BASE_URL}/{endpoint}"
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(1)
        return {}

    def _date_range(self, days_back: int) -> tuple[str, str]:
        """Return ISO-formatted start/end dates."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        fmt = "%Y-%m-%dT%HZ"
        return start.strftime(fmt), end.strftime(fmt)

    # ── public ───────────────────────────────────────────────────────────────

    def get_hourly_demand(
        self,
        region: str = "CAL",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch hourly electricity demand (actual load) for a grid region.

        Args:
            region:    EIA RTO/BA region code (e.g. "CAL", "TEX")
            days_back: How many days of history to pull

        Returns:
            DataFrame with columns: timestamp, region, demand_mwh
        """
        start, end = self._date_range(days_back)
        logger.info(f"Fetching demand for {region} from {start} to {end}")

        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",        # D = demand
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }

        data = self._get("electricity/rto/region-data/data", params)
        records = data.get("response", {}).get("data", [])

        if not records:
            logger.warning(f"No demand data returned for region {region}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.rename(columns={"period": "timestamp", "value": "demand_mwh"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region"] = region
        df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
        df = df[["timestamp", "region", "demand_mwh"]].dropna()

        logger.info(f"✓ Fetched {len(df)} demand records for {region}")
        time.sleep(RATE_LIMIT_SLEEP)
        return df

    def get_generation_mix(
        self,
        region: str = "CAL",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch hourly generation by fuel type (solar, wind, gas, coal, nuclear, etc.)

        Returns:
            DataFrame with columns: timestamp, region, fuel_type, generation_mwh
        """
        start, end = self._date_range(days_back)
        logger.info(f"Fetching generation mix for {region}")

        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }

        data = self._get("electricity/rto/fuel-type-data/data", params)
        records = data.get("response", {}).get("data", [])

        if not records:
            logger.warning(f"No generation mix data for {region}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.rename(columns={
            "period": "timestamp",
            "value": "generation_mwh",
            "fueltype": "fuel_type",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["region"] = region
        df["generation_mwh"] = pd.to_numeric(df["generation_mwh"], errors="coerce")
        df = df[["timestamp", "region", "fuel_type", "generation_mwh"]].dropna()

        logger.info(f"✓ Fetched {len(df)} generation records for {region}")
        time.sleep(RATE_LIMIT_SLEEP)
        return df

    def get_all_regions(
        self,
        regions: list[str] = DEFAULT_REGIONS,
        days_back: int = 30,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch demand + generation mix for multiple regions.

        Returns:
            (demand_df, generation_df) — concatenated across all regions
        """
        demand_dfs, gen_dfs = [], []

        for region in regions:
            try:
                demand_dfs.append(self.get_hourly_demand(region, days_back))
                gen_dfs.append(self.get_generation_mix(region, days_back))
            except Exception as e:
                logger.error(f"Failed to fetch {region}: {e}")

        demand = pd.concat(demand_dfs, ignore_index=True) if demand_dfs else pd.DataFrame()
        generation = pd.concat(gen_dfs, ignore_index=True) if gen_dfs else pd.DataFrame()

        logger.info(
            f"✓ Total: {len(demand)} demand rows, {len(generation)} generation rows "
            f"across {len(regions)} regions"
        )
        return demand, generation


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    client = EIAClient()

    print("\n── Demand (California, last 7 days) ──")
    df = client.get_hourly_demand(region="CAL", days_back=7)
    print(df.head(5).to_string(index=False))
    print(f"Shape: {df.shape}")

    print("\n── Generation Mix (California, last 7 days) ──")
    gen = client.get_generation_mix(region="CAL", days_back=7)
    print(gen.groupby("fuel_type")["generation_mwh"].mean().round(1).sort_values(ascending=False))
