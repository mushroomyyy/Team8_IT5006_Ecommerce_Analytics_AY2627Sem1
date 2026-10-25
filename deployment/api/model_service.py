"""Load and validate the packaged regression model used by the API and UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np

from .feature_validation import validate_features


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "regression_r1_ols.joblib"


class ModelService:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}. "
                "Run `python -m deployment.export.train_regression` first."
            )
        self.bundle = joblib.load(self.artifact_path)
        self.model = self.bundle["model"]
        self.metadata = self.bundle["metadata"]
        # Both models expose the same approval-time domain contract. Regression's
        # expansion merges route categories internally; validate original values.
        self.validation_metadata = {**self.metadata, "categories": json.loads(
            (DEFAULT_ARTIFACT.parent / "classification_model_stats.json").read_text())["categories"]}
        self.feature_cols = self.metadata["features"]["numeric"] + self.metadata["features"]["categorical"]

    def validate_features(self, features: dict[str, Any]) -> pd.DataFrame:
        return validate_features(features, self.validation_metadata, model=self.model)

    def predictions(self, frame: pd.DataFrame):
        values = np.asarray(self.model.predict(frame), dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Regressor returned nonfinite predictions")
        return values

    def format_prediction(self, raw_days: float, features: dict[str, Any]) -> dict[str, Any]:
        promised = features.get("promised_lead_days")
        clipped = promised is not None
        predicted_days = float(np.clip(raw_days, -promised, 45 - promised)) if clipped else raw_days
        return {
            "predicted_days_from_promise": predicted_days,
            "raw_days_from_promise": raw_days,
            "late_by_days": max(predicted_days, 0.0),
            "early_by_days": max(-predicted_days, 0.0),
            "clipping_applied": clipped,
            "clipping_policy": "Predicted approval-to-delivery duration bounded to 0–45 days, matching Phase 2 evaluation." if clipped else "Promise missing: waiting-period clipping unavailable; raw model estimate shown.",
            "target": "days from promised delivery date; negative means early, positive means late",
            "model_code": self.metadata["model_code"],
            "model_version": self.metadata["model_version"],
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self.format_prediction(float(self.predictions(self.validate_features(features))[0]), features)


def load_stats() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "regression_model_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}
