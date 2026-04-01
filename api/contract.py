"""Inference contract for demand forecasting endpoints."""

from __future__ import annotations


REQUIRED_FEATURES = [
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "renewable_lag_24h",
    "demand_roll_mean_24h",
    "demand_roll_std_24h",
    "demand_roll_mean_168h",
    "demand_roll_max_24h",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temperature_2m",
    "heating_degrees",
    "cooling_degrees",
    "temp_x_hour",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
    "renewable_pct",
    "demand_vs_roll_mean",
    "wind_power_potential",
    "effective_solar",
]

INTEGER_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month_of_year",
    "is_weekend",
    "is_peak_hour",
]

TARGET_FIELD = "predicted_demand_mwh"
