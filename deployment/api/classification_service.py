"""Load the selected classification pipeline and run all model work behind the HTTP boundary."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_validation import validate_features

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_ARTIFACT = ARTIFACTS / "classification_model.joblib"


class ClassificationService:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Missing classification artifact: {self.artifact_path}. Run python -m deployment.export.train_classification first.")
        bundle = joblib.load(self.artifact_path)
        self.model, self.metadata = bundle["model"], bundle["metadata"]
        self.feature_cols = self.metadata["features"]["numeric"] + self.metadata["features"]["categorical"]
        self.positive_index = list(self.model.classes_).index(self.metadata["positive_class"])

    def validate_features(self, features: dict[str, Any]):
        return validate_features(features, self.metadata, model=self.model)

    def probabilities(self, frame):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
            values = np.asarray(self.model.predict_proba(frame)[:, self.positive_index])
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError("Classifier returned an invalid probability")
        return values

    def prediction_from_frame(self, frame, features):
        return self.format_prediction(float(self.probabilities(frame)[0]), features)

    def format_prediction(self, probability, features):
        warnings_list = []
        for name, limits in self.metadata.get("numeric_ranges", {}).items():
            value = features.get(name)
            if value is not None and (value < limits["p01"] or value > limits["p99"]):
                warnings_list.append({"feature": name, "value": value, "p01": limits["p01"], "p99": limits["p99"],
                                      "message": "Outside the historical refit 1st–99th percentile; check this input."})
        return {"late_probability": probability, "drift_warnings": warnings_list,
                "score_interpretation": "Estimated late-delivery risk; not a calibrated guarantee.",
                "model_code": self.metadata["model_code"], "model_version": self.metadata["model_version"]}

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self.prediction_from_frame(self.validate_features(features), features)


def load_stats() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "classification_model_stats.json").read_text())
