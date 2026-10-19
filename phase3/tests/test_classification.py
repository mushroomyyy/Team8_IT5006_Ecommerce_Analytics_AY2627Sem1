"""Checks against the actual exported model, not a mocked classifier.

Run: python -m unittest phase3.tests.test_classification
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from phase3.backend.classification_service import ARTIFACTS, ClassificationService, load_stats


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ClassificationService()
        cls.samples = json.loads((ARTIFACTS / "classification_sample_orders.json").read_text())
        cls.features = {key: value for key, value in cls.samples[0].items() if key != "order_id"}

    def test_export_matches_notebook_specification_and_evaluation(self):
        root = Path(__file__).resolve().parents[2]
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
                               run["headline_june"]["C3"]["avg_precision"], places=15)
        metrics = pd.read_csv(root / "phase2/results/simple_to_complex/classification/model_results_long.csv",
                              float_precision="round_trip")
        expected = metrics[metrics["model"].eq("C3")].set_index("split").drop(columns="model").to_dict(orient="index")
        self.assertEqual(metadata["evaluation"], expected)
        self.assertEqual(metadata["fingerprints"]["notebook_normalized"], run["fingerprints"]["C3"])
        self.assertEqual(metadata["fingerprints"]["notebook_reference"], run["fingerprints"]["C3"])

    def test_replay_is_feature_only_and_spans_june(self):
        replay = json.loads((ARTIFACTS / "classification_june_orders.json").read_text())
        outcomes = json.loads((ARTIFACTS / "classification_june_outcomes.json").read_text())
        columns = set(self.service.metadata["features"]["numeric"] + self.service.metadata["features"]["categorical"] + ["order_id"])
        rows = [row for day_rows in replay["days"].values() for row in day_rows]
        self.assertEqual(len(rows), 6164)
        self.assertEqual(len({row["order_id"] for row in rows}), len(rows))
        for row in rows:
            self.assertEqual(set(row), columns)
        self.assertEqual(set(outcomes["outcomes"]), {row["order_id"] for row in rows})
        self.assertTrue(outcomes["display_only"])
        self.assertEqual(sum(value is None for value in outcomes["outcomes"].values()), 9)
        sample_ids = {row["order_id"] for row in self.samples}
        self.assertEqual(len(sample_ids), 30)
        represented_days = [day for day, day_rows in replay["days"].items()
                            if any(row["order_id"] in sample_ids for row in day_rows)]
        self.assertEqual(len(represented_days), 30)

    def test_ranges_baseline_and_score_reference(self):
        meta = self.service.metadata
        for name in meta["features"]["numeric"]:
            ranges = meta["numeric_ranges"][name]
            self.assertLessEqual(ranges["min"], ranges["p01"])
            self.assertLessEqual(ranges["p01"], ranges["median"])
            self.assertLessEqual(ranges["median"], ranges["p99"])
            self.assertLessEqual(ranges["p99"], ranges["max"])
            self.assertIn(ranges["type"], ("integer", "binary", "continuous"))
            self.assertEqual(meta["baseline_features"][name], ranges["median"])
        distribution = meta["june_score_distribution"]
        scores = distribution["sorted_scores"]
        self.assertEqual(len(scores), 6164)
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(sum(bin_["count"] for bin_ in distribution["histogram"]), 6164)
        self.assertEqual(distribution["top_10_percent_boundary"], float(np.quantile(scores, .9)))
        self.assertEqual(set(meta["global_importance"]), {"Validation", "June"})

    def test_service_preserves_full_precision_probabilities(self):
        frame = self.service.validate_features(self.features)
        expected = self.service.model.predict_proba(frame)[0, list(self.service.model.classes_).index(1)]
        result = self.service.predict(self.features)
        self.assertEqual(result["late_probability"], expected)
        self.assertNotIn("is_predicted_late", result)
        self.assertNotIn("recommended_action", result)
        self.assertEqual(result["model_code"], "C3")

    def test_entire_june_scores_preserve_artifact_and_reference(self):
        replay = json.loads((ARTIFACTS / "classification_june_orders.json").read_text())
        rows = [row for orders in replay["days"].values() for row in orders]
        features = [{k: v for k, v in row.items() if k != "order_id"} for row in rows]
        frame = pd.DataFrame(features, columns=self.service.feature_cols)
        actual = self.service.probabilities(frame)
        expected = self.service.model.predict_proba(frame)[:, self.service.positive_index]
        np.testing.assert_array_equal(actual, expected)
        # Export metadata records every June score, so this also detects changed
        # preprocessing, positive-class selection and model artifact regressions.
        np.testing.assert_allclose(np.sort(actual), self.service.metadata["june_score_distribution"]["sorted_scores"], rtol=0, atol=1e-15)

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
