"""Packaged R1 precision, clipping, validation and inference contract checks."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from phase3.backend import api
from phase3.backend.classification_service import ClassificationService
from phase3.backend.model_service import ModelService

DATA = Path(__file__).resolve().parents[1] / "frontend" / "data"


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

    def test_http_batch_single_and_independent_r1_agree(self):
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
