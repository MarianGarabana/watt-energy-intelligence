"""MLflow model loading and inference service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import mlflow
import pandas as pd

from api.contract import INTEGER_FEATURES, REQUIRED_FEATURES


@dataclass
class DemandModelService:
    model_name: str = "demand_forecaster"

    def __post_init__(self) -> None:
        self.model_uri: str | None = None
        self.model = None
        self.signature: str | None = None

    def load_latest_local_model(self, repo_root: Path) -> None:
        """Load the most recently updated local MLflow model artifact."""
        candidates = list(repo_root.rglob("mlartifacts/**/artifacts/MLmodel"))
        if not candidates:
            raise FileNotFoundError("No MLmodel file found under mlartifacts.")

        latest_mlmodel = max(candidates, key=lambda p: p.stat().st_mtime)
        model_dir = latest_mlmodel.parent

        self.model = mlflow.pyfunc.load_model(str(model_dir))
        self.model_uri = str(model_dir)
        if self.model.metadata and self.model.metadata.signature:
            self.signature = str(self.model.metadata.signature)

    def load(self, repo_root: Path) -> None:
        """Load model from MODEL_URI env var or fallback to local artifacts."""
        configured_uri = os.getenv("WATT_MODEL_URI") or os.getenv("MODEL_URI")
        if configured_uri:
            self.model = mlflow.pyfunc.load_model(configured_uri)
            self.model_uri = configured_uri
            if self.model.metadata and self.model.metadata.signature:
                self.signature = str(self.model.metadata.signature)
            return

        self.load_latest_local_model(repo_root=repo_root)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_one(self, features: dict[str, float | int]) -> float:
        if not self.model:
            raise RuntimeError("Model is not loaded.")

        row = {key: features[key] for key in REQUIRED_FEATURES}
        for key in INTEGER_FEATURES:
            row[key] = int(row[key])
        frame = pd.DataFrame([row])
        prediction = self.model.predict(frame)
        return float(prediction[0])
