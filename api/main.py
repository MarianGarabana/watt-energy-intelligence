"""FastAPI entrypoint for WATT demand forecasting endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.contract import REQUIRED_FEATURES, TARGET_FIELD
from api.model_service import DemandModelService
from api.schemas import (
    DemandPredictionRequest,
    DemandPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


app = FastAPI(
    title="WATT Energy Intelligence API",
    description="Inference API for electricity demand forecasting.",
    version="0.1.0",
)
service = DemandModelService()


@app.on_event("startup")
def startup_load_model() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        service.load(repo_root=repo_root)
    except Exception:
        # Keep app alive for debugging and non-inference endpoints.
        service.model = None
        service.model_uri = None
        service.signature = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_uri=service.model_uri,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelInfoResponse(
        model_name=service.model_name,
        model_uri=service.model_uri or "",
        required_features=REQUIRED_FEATURES,
        output_field=TARGET_FIELD,
        signature=service.signature,
        extras={"n_required_features": len(REQUIRED_FEATURES)},
    )


@app.post("/predict/demand", response_model=DemandPredictionResponse)
def predict_demand(payload: DemandPredictionRequest) -> DemandPredictionResponse:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        predicted = service.predict_one(payload.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return DemandPredictionResponse.build(
        model_name=service.model_name,
        model_uri=service.model_uri or "",
        predicted_value=predicted,
    )
