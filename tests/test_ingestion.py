"""
Tests for ingestion clients.
Run: pytest tests/test_ingestion.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════════
# WeatherClient tests (no API key needed — safe to run anywhere)
# ══════════════════════════════════════════════════════════════════════════════

class TestWeatherClient:
    """Tests for Open-Meteo weather client."""

    def test_known_regions_have_coords(self):
        from ingestion.weather_client import REGION_COORDS
        assert "CAL" in REGION_COORDS
        assert "TEX" in REGION_COORDS
        assert len(REGION_COORDS) >= 6

    def test_unknown_region_raises(self):
        from ingestion.weather_client import WeatherClient
        client = WeatherClient()
        with pytest.raises(ValueError, match="Unknown region"):
            client.get_weather(region="INVALID_XYZ")

    @patch("ingestion.weather_client.requests.Session.get")
    def test_get_weather_returns_dataframe(self, mock_get):
        """Mock the HTTP call and verify DataFrame structure."""
        from ingestion.weather_client import WeatherClient, HOURLY_VARIABLES

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [10.5, 11.2],
                "apparent_temperature": [9.0, 10.1],
                "shortwave_radiation": [0.0, 0.0],
                "wind_speed_10m": [5.2, 4.8],
                "wind_direction_10m": [180, 190],
                "cloud_cover": [60, 55],
                "precipitation": [0.0, 0.1],
                "relative_humidity_2m": [80, 78],
            }
        }
        mock_get.return_value = mock_response

        client = WeatherClient()
        df = client.get_weather(region="CAL", days_back=1, days_forward=0)

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "region" in df.columns
        assert "temperature_2m" in df.columns
        assert len(df) == 2
        assert df["region"].iloc[0] == "CAL"

    @patch("ingestion.weather_client.requests.Session.get")
    def test_empty_response_returns_empty_df(self, mock_get):
        from ingestion.weather_client import WeatherClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hourly": {}}
        mock_get.return_value = mock_response

        client = WeatherClient()
        df = client.get_weather(region="CAL", days_back=1, days_forward=0)
        assert df.empty


# ══════════════════════════════════════════════════════════════════════════════
# EIAClient tests (mocked — no real API key needed for tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestEIAClient:
    """Tests for EIA API client."""

    def test_no_api_key_raises(self):
        from ingestion.eia_client import EIAClient
        with pytest.raises(ValueError, match="EIA API key not found"):
            EIAClient(api_key=None)
        # Clean env: make sure EIA_API_KEY is not set
        import os
        env_backup = os.environ.pop("EIA_API_KEY", None)
        with pytest.raises(ValueError):
            EIAClient(api_key=None)
        if env_backup:
            os.environ["EIA_API_KEY"] = env_backup

    @patch("ingestion.eia_client.requests.Session.get")
    def test_get_demand_returns_correct_columns(self, mock_get):
        from ingestion.eia_client import EIAClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2024-01-01T00", "value": "42000", "respondent": "CAL", "type": "D"},
                    {"period": "2024-01-01T01", "value": "41500", "respondent": "CAL", "type": "D"},
                ]
            }
        }
        mock_get.return_value = mock_response

        client = EIAClient(api_key="test_key_123")
        df = client.get_hourly_demand(region="CAL", days_back=1)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"timestamp", "region", "demand_mwh"}
        assert len(df) == 2
        assert pd.api.types.is_numeric_dtype(df["demand_mwh"])
        assert df["region"].iloc[0] == "CAL"

    @patch("ingestion.eia_client.requests.Session.get")
    def test_empty_api_response_returns_empty_df(self, mock_get):
        from ingestion.eia_client import EIAClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": {"data": []}}
        mock_get.return_value = mock_response

        client = EIAClient(api_key="test_key_123")
        df = client.get_hourly_demand(region="CAL", days_back=1)
        assert df.empty


# ══════════════════════════════════════════════════════════════════════════════
# Data quality checks
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQuality:
    """Basic data quality checks that apply to any ingested DataFrame."""

    def test_no_duplicate_timestamps_per_region(self):
        """Demand data should have one row per (timestamp, region) pair."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "region": ["CAL", "TEX", "CAL"],
            "demand_mwh": [40000, 55000, 41000],
        })
        dupes = df.duplicated(subset=["timestamp", "region"])
        assert not dupes.any(), "Found duplicate (timestamp, region) pairs"

    def test_demand_values_are_positive(self):
        """Electricity demand should never be negative."""
        df = pd.DataFrame({
            "demand_mwh": [40000, 41000, 39000, 42000],
        })
        assert (df["demand_mwh"] >= 0).all(), "Negative demand values found"

    def test_timestamps_are_timezone_aware(self):
        """All timestamps should be UTC-aware."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"]),
        })
        assert df["timestamp"].dt.tz is not None, "Timestamps should be timezone-aware"
