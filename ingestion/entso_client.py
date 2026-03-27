"""
ENTSO-E Transparency Platform API Client
==========================================
Fetches European electricity grid data: actual load, generation per source,
cross-border flows — for 35+ countries.

Free API key: https://transparency.entsoe.eu/ (register → request token)

Country codes (EIC area codes):
  DE  - Germany     (10Y1001A1001A83F)
  FR  - France      (10YFR-RTE------C)
  ES  - Spain       (10YES-REE------0)
  IT  - Italy       (10YIT-GRTN-----B)
  UK  - UK          (10YGB----------A)
  NL  - Netherlands (10YNL----------L)
  PT  - Portugal    (10YPT-REN------W)
  PL  - Poland      (10YPL-AREA-----S)
"""

import os
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ENTSO_BASE_URL = "https://web-api.tp.entsoe.eu/api"
REQUEST_TIMEOUT = 30
RATE_LIMIT_SLEEP = 1.0

# Country → EIC area code
COUNTRY_CODES = {
    "DE": "10Y1001A1001A83F",
    "FR": "10YFR-RTE------C",
    "ES": "10YES-REE------0",
    "IT": "10YIT-GRTN-----B",
    "UK": "10YGB----------A",
    "NL": "10YNL----------L",
    "PT": "10YPT-REN------W",
    "PL": "10YPL-AREA-----S",
}

# ENTSO-E document type codes
DOC_TYPE_LOAD       = "A65"   # Actual Total Load
DOC_TYPE_GENERATION = "A75"   # Actual Generation Per Type
PROCESS_TYPE        = "A16"   # Realised

# XML namespace used in ENTSO-E responses
NS = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}


class ENTSOClient:
    """
    Client for the ENTSO-E Transparency Platform REST API.

    Usage:
        client = ENTSOClient()
        df = client.get_actual_load(country="DE", days_back=7)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ENTSO-E API key not found. Set ENTSO_API_KEY in your .env file.\n"
                "Register at: https://transparency.entsoe.eu/"
            )

    # ── private ──────────────────────────────────────────────────────────────

    def _get(self, params: dict) -> str:
        """Make a GET request and return raw XML response text."""
        params["securityToken"] = self.api_key
        for attempt in range(3):
            try:
                r = requests.get(ENTSO_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.text
            except requests.HTTPError as e:
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1)
        return ""

    def _date_params(self, days_back: int) -> tuple[str, str]:
        """Return ENTSO-E formatted date strings: YYYYMMDDHHММ."""
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days_back)
        fmt = "%Y%m%d%H%M"
        return start.strftime(fmt), end.strftime(fmt)

    def _parse_load_xml(self, xml_text: str, country: str) -> pd.DataFrame:
        """Parse ENTSO-E Actual Load XML response into a DataFrame."""
        if not xml_text:
            return pd.DataFrame()

        root = ET.fromstring(xml_text)
        records = []

        for ts in root.findall(".//ns:TimeSeries", NS):
            period = ts.find("ns:Period", NS)
            if period is None:
                continue

            start_el = period.find("ns:timeInterval/ns:start", NS)
            resolution_el = period.find("ns:resolution", NS)
            if start_el is None or resolution_el is None:
                continue

            period_start = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))
            resolution_str = resolution_el.text  # e.g. "PT60M"
            minutes = int("".join(filter(str.isdigit, resolution_str.split("T")[-1].replace("M", ""))))
            delta = timedelta(minutes=minutes)

            for point in period.findall("ns:Point", NS):
                pos = int(point.find("ns:position", NS).text)
                qty = point.find("ns:quantity", NS)
                if qty is None:
                    continue
                ts_val = period_start + delta * (pos - 1)
                records.append({
                    "timestamp": ts_val,
                    "country": country,
                    "load_mw": float(qty.text),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    # ── public ───────────────────────────────────────────────────────────────

    def get_actual_load(
        self,
        country: str = "DE",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch actual total electricity load (demand) for a European country.

        Args:
            country:   Country code (e.g. "DE", "FR", "ES")
            days_back: How many days of history to pull (max ~30 per call)

        Returns:
            DataFrame with columns: timestamp, country, load_mw
        """
        if country not in COUNTRY_CODES:
            raise ValueError(f"Unknown country '{country}'. Available: {list(COUNTRY_CODES.keys())}")

        area_code = COUNTRY_CODES[country]
        start, end = self._date_params(days_back)

        logger.info(f"Fetching ENTSO-E load for {country}: {start} → {end}")

        params = {
            "documentType": DOC_TYPE_LOAD,
            "processType": PROCESS_TYPE,
            "outBiddingZone_Domain": area_code,
            "periodStart": start,
            "periodEnd": end,
        }

        xml_text = self._get(params)
        df = self._parse_load_xml(xml_text, country)

        if df.empty:
            logger.warning(f"No load data returned for {country}")
        else:
            logger.info(f"✓ {len(df)} load records for {country}")

        time.sleep(RATE_LIMIT_SLEEP)
        return df

    def get_all_countries(
        self,
        countries: list[str] = ["DE", "FR", "ES"],
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Fetch load data for multiple European countries."""
        dfs = []
        for country in countries:
            try:
                dfs.append(self.get_actual_load(country, days_back))
            except Exception as e:
                logger.error(f"ENTSO-E fetch failed for {country}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✓ Total ENTSO-E records: {len(df)} across {len(countries)} countries")
        return df


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    client = ENTSOClient()

    print("\n── Actual Load (Germany, last 7 days) ──")
    df = client.get_actual_load(country="DE", days_back=7)
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
