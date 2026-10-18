"""Load the complete C3 preprocessing/classification artifact for inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_validation import validate_features

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ARTIFACT = ARTIFACTS / "classification_c3_rf.joblib"


class ClassificationService:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Missing classification artifact: {self.artifact_path}. "
                                    "Run python -m deployment.train_classification first.")
        bundle = joblib.load(self.artifact_path)
        self.model, self.metadata = bundle["model"], bundle["metadata"]
        self.feature_cols = self.metadata["features"]["numeric"] + self.metadata["features"]["categorical"]
        self.positive_index = list(self.model.classes_).index(self.metadata["positive_class"])
        self.threshold = float(self.metadata["threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("Artifact threshold must be between zero and one")

    def validate_features(self, features: dict[str, Any]):
        return validate_features(features, self.metadata, model=self.model)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        frame = self.validate_features(features)
        probability = float(self.model.predict_proba(frame)[0, self.positive_index])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Classifier returned an invalid probability")
        late = probability >= self.threshold
        return {"late_probability": probability, "is_predicted_late": late,
                "threshold": self.threshold, "threshold_policy": self.metadata["threshold_policy"],
                "delivery_status": "late" if late else "before_or_on_promise",
                "recommended_action": "review" if late else "monitor",
                "model_code": self.metadata["model_code"], "model_version": self.metadata["model_version"]}


def load_stats() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "classification_model_stats.json").read_text())
