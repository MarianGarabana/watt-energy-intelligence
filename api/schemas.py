"""Pydantic schemas for API request/response payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.contract import REQUIRED_FEATURES, TARGET_FIELD


class DemandPredictionRequest(BaseModel):
    """Single-row inference request for demand forecasting."""

    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | int] = Field(
        ...,
        description="Feature map used for one demand prediction row.",
    )

    @model_validator(mode="after")
    def validate_feature_keys(self) -> "DemandPredictionRequest":
        provided = set(self.features.keys())
        required = set(REQUIRED_FEATURES)
        missing = sorted(required - provided)
        extra = sorted(provided - required)
        if missing or extra:
            raise ValueError(
                f"Feature mismatch. Missing={missing or []}; Extra={extra or []}"
            )
        return self


class DemandPredictionResponse(BaseModel):
    """Response payload for demand prediction endpoint."""

    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_uri: str
    prediction: dict[str, float]

    @staticmethod
    def build(model_name: str, model_uri: str, predicted_value: float) -> "DemandPredictionResponse":
        return DemandPredictionResponse(
            model_name=model_name,
            model_uri=model_uri,
            prediction={TARGET_FIELD: float(predicted_value)},
        )


class HealthResponse(BaseModel):
    """Basic liveness and readiness information."""

    status: str
    model_loaded: bool
    model_uri: str | None = None


class ModelInfoResponse(BaseModel):
    """Model metadata and input/output contract details."""

    model_name: str
    model_uri: str
    required_features: list[str]
    output_field: str
    signature: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)
