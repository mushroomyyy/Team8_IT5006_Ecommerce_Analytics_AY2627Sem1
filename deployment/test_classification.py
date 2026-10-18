"""Checks against the actual exported model, not a mocked classifier.

Run: python -m unittest deployment.test_classification
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from deployment.classification_service import ARTIFACTS, ClassificationService, load_stats


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ClassificationService()
        cls.samples = json.loads((ARTIFACTS / "classification_sample_orders.json").read_text())
        cls.features = {key: value for key, value in cls.samples[0].items() if key != "order_id"}

    def test_export_matches_notebook_specification_and_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        run = json.loads((root / "phase2/results/simple_to_complex/classification/run_metadata.json").read_text())
        metadata = self.service.metadata
        self.assertEqual(metadata, load_stats())
        self.assertEqual(metadata["train_rows"], run["row_counts"]["history"])
        self.assertEqual(metadata["best_params"], run["rf_best_params"])
        actual = self.service.model.named_steps["classifier"].get_params()
        for name, value in metadata["best_params"].items():
            self.assertEqual(actual[name], value)
        self.assertEqual(actual["random_state"], 42)
        self.assertEqual(metadata["features"], run["features"])
        self.assertEqual(metadata["export_verification"]["sample_training_overlap"], 0)
        self.assertAlmostEqual(metadata["export_verification"]["june_ap"],
                               run["headline_june"]["C3"]["avg_precision"], places=10)
        self.assertAlmostEqual(metadata["evaluation"]["June"]["avg_precision"],
                               run["headline_june"]["C3"]["avg_precision"], places=9)

    def test_service_preserves_full_precision_probabilities(self):
        frame = self.service.validate_features(self.features)
        expected = self.service.model.predict_proba(frame)[0, list(self.service.model.classes_).index(1)]
        result = self.service.predict(self.features)
        self.assertEqual(result["late_probability"], expected)
        self.assertEqual(result["is_predicted_late"], expected >= 0.5)
        self.assertIn("not optimized", result["threshold_policy"])
        self.assertEqual(result["model_code"], "C3")

    def test_optional_missing_numeric_is_imputed(self):
        row = {**self.features, "payment_value_sum": None}
        result = self.service.predict(row)
        self.assertTrue(0 <= result["late_probability"] <= 1)

    def test_invalid_values_fail_before_prediction(self):
        cases = [("payment_value_sum", "not a number"), ("payment_value_sum", "100"),
                 ("payment_value_sum", np.inf), ("payment_value_sum", np.nan),
                 ("payment_value_sum", -1), ("payment_value_sum", True),
                 ("order_approved_day_of_week", 0), ("order_approved_day_of_week", 8),
                 ("order_approved_month", 13), ("order_item_count", 1.5),
                 ("is_weekend_approval", 2), ("customer_state", 42)]
        for feature, value in cases:
            with self.subTest(feature=feature, value=value), self.assertRaises(ValueError):
                self.service.predict({**self.features, feature: value})
        with self.assertRaises(ValueError):
            self.service.predict({**self.features, "order_status": "delivered"})
        with self.assertRaises(ValueError):
            self.service.predict({k: v for k, v in self.features.items() if k != "customer_state"})
        with self.assertRaises(ValueError):
            self.service.predict({k: None if k in self.service.metadata["features"]["numeric"] else v
                                  for k, v in self.features.items()})

    def test_probability_uses_documented_positive_class(self):
        # Deliberately reverse classifier columns to ensure the service locates class 1.
        service = ClassificationService()
        original = service.model

        class ReversedClasses:
            classes_ = np.array([1, 0])

            def predict_proba(self, frame):
                return original.predict_proba(frame)[:, ::-1]

        service.model = ReversedClasses()
        service.positive_index = list(service.model.classes_).index(1)
        self.assertEqual(service.predict(self.features)["late_probability"],
                         self.service.predict(self.features)["late_probability"])

    def test_missing_artifact_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "train_classification"):
            ClassificationService(ARTIFACTS / "does-not-exist.joblib")


if __name__ == "__main__":
    unittest.main()
