"""Load the C3 pipeline and run all model work behind the HTTP boundary."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_validation import validate_features

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_ARTIFACT = ARTIFACTS / "classification_c3_rf.joblib"


class ClassificationService:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Missing classification artifact: {self.artifact_path}. Run python -m phase3.export.train_classification first.")
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

    def explain(self, features):
        frame = self.validate_features(features)
        original = float(self.probabilities(frame)[0])
        baselines = self.metadata["baseline_features"]
        # Independent one-feature perturbations intentionally preserve all other
        # inputs. They are sensitivity probes, not valid alternative order dates.
        rows = []
        for name in self.feature_cols:
            row = frame.iloc[0].to_dict()
            row[name] = baselines[name]
            rows.append(row)
        changed = pd.DataFrame(rows, columns=self.feature_cols)
        changed[self.metadata["features"]["categorical"]] = changed[self.metadata["features"]["categorical"]].astype(object)
        scores = self.probabilities(changed)
        effects = [{"feature": name, "baseline_value": baselines[name], "reset_probability": float(score),
                    "probability_change": original - float(score)} for name, score in zip(self.feature_cols, scores)]
        effects.sort(key=lambda item: abs(item["probability_change"]), reverse=True)
        return {"late_probability": original, "sensitivities": effects,
                "method": "One-feature reset to refit numeric median or categorical mode; sensitivity only, not causal. Perturbed rows may break calendar relationships.",
                "model_version": self.metadata["model_version"]}


def load_stats() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "classification_model_stats.json").read_text())
