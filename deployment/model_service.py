"""Load and validate the packaged regression model used by the API and UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "regression_r1_ols.joblib"


class ModelService:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}. "
                "Run `python -m deployment.train_regression` first."
            )
        self.bundle = joblib.load(self.artifact_path)
        self.model = self.bundle["model"]
        self.metadata = self.bundle["metadata"]
        self.feature_cols = self.metadata["features"]["numeric"] + self.metadata["features"]["categorical"]

    def validate_features(self, features: dict[str, Any]) -> pd.DataFrame:
        missing = [column for column in self.feature_cols if column not in features]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        row = {column: features[column] for column in self.feature_cols}
        frame = pd.DataFrame([row])
        for column in self.metadata["features"]["numeric"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[self.metadata["features"]["numeric"]].isna().all(axis=None):
            raise ValueError("At least one numeric model feature must be supplied.")
        return frame

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        frame = self.validate_features(features)
        predicted_days = float(self.model.predict(frame)[0])
        predicted_late = predicted_days > 0
        return {
            "predicted_days_from_promise": round(predicted_days, 3),
            "is_predicted_late": predicted_late,
            "late_by_days": round(max(predicted_days, 0.0), 3),
            "early_by_days": round(max(-predicted_days, 0.0), 3),
            "delivery_status": "late" if predicted_late else "before_or_on_promise",
            "recommended_action": "follow_up" if predicted_late else "monitor",
            "model_code": self.metadata["model_code"],
            "model_version": self.metadata["model_version"],
        }


def load_stats() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "artifacts" / "regression_model_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}
