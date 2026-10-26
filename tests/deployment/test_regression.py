"""Packaged regression model precision, clipping, validation and inference contract checks."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from deployment.api import api
from deployment.api.classification_service import ClassificationService
from deployment.api.model_service import ModelService

DATA = Path(__file__).resolve().parents[2] / "deployment/app/data"


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ModelService()
        cls.features = json.loads((DATA / "classification_sample_orders.json").read_text())[0]
        cls.features = {k: v for k, v in cls.features.items() if k != "order_id"}

    def test_precision_and_phase2_clipping(self):
        model = self.model
        raw = float(model.model.predict(model.validate_features(self.features))[0])
        result = model.predict(self.features)
        self.assertEqual(result["raw_days_from_promise"], raw)
        promise = self.features["promised_lead_days"]
        self.assertEqual(result["predicted_days_from_promise"], float(np.clip(raw, -promise, 45-promise)))
        for raw, expected in ((-100., -promise), (100., 45-promise)):
            self.assertEqual(model.format_prediction(raw, self.features)["predicted_days_from_promise"], expected)
        result = model.predict({**self.features, "promised_lead_days": None})
        self.assertFalse(result["clipping_applied"])
        self.assertEqual(result["predicted_days_from_promise"], result["raw_days_from_promise"])

    def test_model_code_and_run_metadata_come_from_the_notebook(self):
        root = Path(__file__).resolve().parents[2]
        run = json.loads((root / "results/simple_to_complex/regression/run_metadata.json").read_text())
        code = run["selected_model"]
        metadata = self.model.metadata
        self.assertEqual(metadata["model_code"], code)
        self.assertEqual(self.model.predict(self.features)["model_code"], code)
        self.assertEqual(metadata["fingerprints"]["artifact"], run["fingerprints"][code])
        self.assertAlmostEqual(metadata["export_verification"]["june_rmse"], run["headline_june"][code]["rmse"], places=10)
        self.assertEqual(metadata["train_rows"], run["row_counts"]["history"])
        self.assertEqual([row["Code"] for row in metadata["selection"] if row["selected"]], [code])

    def test_subset_model_accepts_and_ignores_the_full_feature_payload(self):
        selected = set(self.model.metadata["selected_features"])
        all_features = set(self.model.feature_cols)
        self.assertEqual(set(self.features), all_features)
        ignored = sorted(all_features - selected)
        if not ignored:
            self.skipTest("The selected model uses every feature")
        numeric_ranges = ClassificationService().metadata["numeric_ranges"]
        continuous = [name for name in ignored if numeric_ranges.get(name, {}).get("type") == "continuous"]
        self.assertTrue(continuous, "Expected an ignored continuous feature to perturb")
        base = self.model.predict(self.features)["raw_days_from_promise"]
        perturbed = {**self.features, continuous[0]: numeric_ranges[continuous[0]]["p99"]}
        self.assertNotEqual(perturbed[continuous[0]], self.features[continuous[0]])
        self.assertEqual(self.model.predict(perturbed)["raw_days_from_promise"], base)
        # A selected feature does move the prediction.
        moved = {**self.features, "promised_lead_days": self.features["promised_lead_days"] + 5}
        self.assertNotEqual(self.model.predict(moved)["raw_days_from_promise"], base)

    def test_both_services_share_strict_validation(self):
        for model in (self.model, ClassificationService()):
            for override in ({"payment_value_sum": "10"}, {"customer_state": None}, {"order_status": "delivered"}, {"is_weekend_approval": 2}):
                with self.subTest(model=type(model).__name__, override=override), self.assertRaises(ValueError):
                    model.predict({**self.features, **override})

    def test_all_sample_batch_predictions_preserve_model_and_single_outputs(self):
        samples = json.loads((DATA / "classification_sample_orders.json").read_text())
        rows = [{k: v for k, v in row.items() if k != "order_id"} for row in samples]
        frame = pd.concat([self.model.validate_features(row) for row in rows], ignore_index=True)
        raw = self.model.predictions(frame)
        np.testing.assert_array_equal(raw, self.model.model.predict(frame))
        for features, batch_raw in zip(rows, raw):
            batch = self.model.format_prediction(float(batch_raw), features)
            single = self.model.predict(features)
            for key in ("raw_days_from_promise", "predicted_days_from_promise"):
                self.assertAlmostEqual(batch[key], single[key], delta=1e-12)
            duration = batch["predicted_days_from_promise"] + features["promised_lead_days"]
            self.assertGreaterEqual(duration, 0)
            self.assertLessEqual(duration, 45)

    def test_http_batch_single_and_independent_regression_agree(self):
        with TestClient(api.app) as client:
            single = client.post("/predict", json={"features": self.features})
            independent = client.post("/predict/regression", json={"features": self.features})
            batch = client.post("/predict/batch", json={"rows": [{"features": self.features},
                {"features": {**self.features, "promised_lead_days": None}}]})
            self.assertEqual(single.status_code, 200)
            self.assertEqual(independent.status_code, 200)
            self.assertEqual(batch.status_code, 200)
            self.assertEqual(batch.json()["successful_rows"], 2)
            self.assertEqual(single.json()["regression"], independent.json())
            for key in ("raw_days_from_promise", "predicted_days_from_promise"):
                self.assertAlmostEqual(single.json()["regression"][key],
                    batch.json()["results"][0]["prediction"]["regression"][key], delta=1e-12)
            missing = batch.json()["results"][1]["prediction"]["regression"]
            self.assertFalse(missing["clipping_applied"])
            self.assertEqual(missing["raw_days_from_promise"], missing["predicted_days_from_promise"])


if __name__ == "__main__":
    unittest.main()
