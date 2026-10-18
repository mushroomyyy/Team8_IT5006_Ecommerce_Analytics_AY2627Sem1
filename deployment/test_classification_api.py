"""Contract and input-validation tests independent of a trained artifact."""

from __future__ import annotations

import unittest
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from deployment import classification_api as api
from deployment.feature_validation import validate_features, FeatureValidationError
from deployment.classification_service import DEFAULT_ARTIFACT


METADATA = {
    "model_code": "C3",
    "model_version": "test-model",
    "categories": {"customer_state": ["SP", "RJ"]},
    "features": {
        "numeric": ["payment_value_sum", "order_approved_month", "order_approved_day_of_week", "order_item_count", "is_weekend_approval"],
        "categorical": ["customer_state"],
    },
}
FEATURES = {
    "payment_value_sum": 120.0,
    "order_approved_month": 6,
    "order_approved_day_of_week": 5,
    "order_item_count": 1,
    "is_weekend_approval": 0,
    "customer_state": "SP",
}


class FakeService:
    metadata = METADATA

    def validate_features(self, features):
        return validate_features(features, self.metadata)

    def probabilities(self, frame):
        return np.full(len(frame), 0.12)

    def format_prediction(self, probability, features):
        return {"late_probability": probability, "is_predicted_late": False, "model_code": "C3", "model_version": "test-model"}

    def predict(self, features):
        validate_features(features, self.metadata)
        return {"late_probability": 0.12, "is_predicted_late": False,
                "model_code": "C3", "model_version": "test-model"}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.factory = patch.object(api, "ClassificationService", FakeService)
        self.factory.start()
        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.factory.stop()

    def test_prediction_and_alias_return_classifier_result(self):
        for endpoint in ("/predict", "/predict/classification"):
            response = self.client.post(endpoint, json={"features": FEATURES})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model_code"], "C3")
            self.assertEqual(response.json()["late_probability"], 0.12)

    def test_health_metadata_and_swagger_are_available(self):
        self.assertEqual(self.client.get("/health").json()["model_loaded"], True)
        for endpoint in ("/model-info", "/model-info/classification"):
            self.assertEqual(self.client.get(endpoint).json(), METADATA)
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertIn("/predict", self.client.get("/openapi.json").json()["paths"])

    def test_bad_request_shape_returns_422(self):
        for payload in ({}, {"features": []}, {"features": FEATURES, "unexpected": 1}):
            self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_bad_features_return_422(self):
        missing = dict(FEATURES)
        del missing["customer_state"]
        invalid = [missing, {**FEATURES, "outcome": 1}]
        invalid += [{**FEATURES, "payment_value_sum": value} for value in ("120", "not a number", True, -1)]
        invalid += [{**FEATURES, "order_approved_month": value} for value in (0, 13, 1.5)]
        invalid += [{**FEATURES, "order_approved_day_of_week": value} for value in (0, 8, 1.5)]
        invalid += [{**FEATURES, "order_item_count": 1.5},
                    {**FEATURES, "is_weekend_approval": 2},
                    {**FEATURES, "customer_state": 100},
                    {**FEATURES, "customer_state": " "}]
        for features in invalid:
            with self.subTest(features=features):
                response = self.client.post("/predict", json={"features": features})
                self.assertEqual(response.status_code, 422, response.text)

    def test_numeric_missing_allowed_but_unknown_or_null_categories_rejected(self):
        self.assertEqual(self.client.post("/predict", json={"features": {**FEATURES, "payment_value_sum": None}}).status_code, 200)
        for value in (None, "unseen_state", "rj", "Sao Paulo"):
            response = self.client.post("/predict", json={"features": {**FEATURES, "customer_state": value}})
            self.assertEqual(response.status_code, 422)
            self.assertIn("SP, RJ", response.text)

    def test_unexpected_model_failure_is_generic_500(self):
        with patch.object(FakeService, "predict", side_effect=ValueError("private model internals")):
            with self.assertLogs(api.logger, level="ERROR"):
                response = self.client.post("/predict", json={"features": FEATURES})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private", response.text)

    def test_batch_limits_row_errors_and_monitoring_privacy(self):
        rows = [{"order_id": "private-order-id", "features": FEATURES},
                {"features": {**FEATURES, "customer_state": None}}, {"bad": 1}]
        result = self.client.post("/predict/batch", json={"rows": rows})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["successful_rows"], 1)
        self.assertEqual(result.json()["results"][1]["status_code"], 422)
        for payload in ([], rows * 334):
            self.assertEqual(self.client.post("/predict/batch", json={"rows": payload}).status_code, 422)
        summary = self.client.get("/logs/summary")
        self.assertEqual(summary.json()["request_count"], 1)
        self.assertNotIn("private-order-id", summary.text)
        self.assertNotIn("payment_value_sum", summary.text)
        self.assertEqual(summary.json()["window_capacity"], 1000)

    def test_monitoring_is_bounded_and_thread_safe(self):
        result = {"late_probability": 0.12, "above_watch_list_boundary": True,
                  "drift_warnings": [{"feature": "payment_value_sum", "value": "never stored"}]}
        def record(_):
            api.record_request()
            api.record_prediction(result)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(record, range(1100)))
        summary = self.client.get("/logs/summary").json()
        self.assertEqual(summary["prediction_count"], 1100)
        self.assertEqual(summary["request_count"], 1100)
        self.assertEqual(summary["window_size"], 1000)
        self.assertEqual(summary["share_above_watch_list_boundary"], 1)
        self.assertEqual(len(summary["recent_warnings"]), 20)
        self.assertNotIn("never stored", str(summary))

    def test_calendar_flags_counts_and_sanity_bound(self):
        numeric = METADATA["features"]["numeric"] + ["order_approved_day_of_month", "is_december_peak", "payment_installments_max"]
        meta = {**METADATA, "features": {"numeric": numeric, "categorical": ["customer_state"]}}
        valid = {**FEATURES, "order_approved_day_of_month": 15, "is_december_peak": 0, "payment_installments_max": 1}
        for override in ({"order_approved_month": 2, "order_approved_day_of_month": 31},
                         {"order_item_count": 0}, {"payment_installments_max": 0},
                         {"is_weekend_approval": 1}, {"is_december_peak": 1}, {"payment_value_sum": 1e39}):
            with self.subTest(override=override), self.assertRaises(FeatureValidationError):
                validate_features({**valid, **override}, meta)
        validate_features({**valid, "order_approved_month": 2, "order_approved_day_of_month": 29}, meta)
        validate_features({**valid, "order_approved_month": None, "is_december_peak": None}, meta)

    def test_all_numeric_missing_rejected(self):
        features = {**FEATURES, **dict.fromkeys(METADATA["features"]["numeric"])}
        self.assertEqual(self.client.post("/predict", json={"features": features}).status_code, 422)

    def test_nonfinite_numbers_rejected_before_inference(self):
        for value in (float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.assertRaisesRegex(ValueError, "finite number"):
                validate_features({**FEATURES, "payment_value_sum": value}, METADATA)

    def test_missing_values_are_normalized_for_fitted_imputer(self):
        frame = validate_features({**FEATURES, "payment_value_sum": None}, METADATA)
        self.assertTrue(np.isnan(frame.loc[0, "payment_value_sum"]))
        self.assertEqual(frame.loc[0, "customer_state"], "SP")
        self.assertEqual(frame["customer_state"].dtype, object)
        self.assertEqual(list(frame), METADATA["features"]["numeric"] + METADATA["features"]["categorical"])


class UnavailableTests(unittest.TestCase):
    def test_failed_load_exposes_unavailable_health_and_predictions(self):
        with patch.object(api, "ClassificationService", side_effect=FileNotFoundError("missing artifact")):
            with self.assertLogs(api.logger, level="ERROR"):
                with TestClient(api.app) as client:
                    for endpoint in ("/health", "/model-info"):
                        self.assertEqual(client.get(endpoint).status_code, 503)
                    self.assertEqual(client.post("/predict", json={"features": FEATURES}).status_code, 503)


@unittest.skipUnless(DEFAULT_ARTIFACT.exists(), "Train the classification artifact to run integration checks")
class PackagedModelApiTests(unittest.TestCase):
    def test_real_samples_replay_batch_explain_and_drift(self):
        with TestClient(api.app) as client:
            metadata = client.get("/model-info").json()
            samples = client.get("/samples").json()
            self.assertEqual(samples["model_version"], metadata["model_version"])
            listing = client.get("/replay").json()
            self.assertGreaterEqual(len(listing["available_days"]), 28)
            self.assertEqual(listing["orders"], [])
            day = listing["available_days"][0]
            replay = client.get("/replay", params={"day": day}).json()
            self.assertNotIn("outcome", str(replay))
            rows = [{"order_id": row["order_id"], "features": {k: v for k, v in row.items() if k != "order_id"}}
                    for row in replay["orders"]]
            batch = client.post("/predict/batch", json={"rows": rows}).json()
            self.assertEqual(batch["successful_rows"], len(rows))
            self.assertTrue(all(0 <= row["prediction"]["late_probability"] <= 1 for row in batch["results"]))
            outcomes = client.get("/outcomes", params={"day": day}).json()
            self.assertTrue(outcomes["display_only"])
            self.assertEqual(set(outcomes["outcomes"]), {row["order_id"] for row in rows})
            self.assertEqual(client.get("/replay", params={"day": "2018-07-01"}).status_code, 422)
            self.assertEqual(client.get("/outcomes", params={"day": "bad"}).status_code, 422)
            features = rows[0]["features"]
            explanation = client.post("/explain", json={"features": features})
            self.assertEqual(explanation.status_code, 200, explanation.text)
            effects = explanation.json()["sensitivities"]
            self.assertEqual(len(effects), 24)
            self.assertEqual({item["feature"] for item in effects}, set(features))
            self.assertEqual(client.get("/logs/summary").json()["prediction_count"], len(rows))
            self.assertEqual(client.get("/logs/summary").json()["request_count"], 1)
            drift = {**features, "payment_value_sum": 1e8}
            prediction = client.post("/predict", json={"features": drift}).json()
            self.assertIn("payment_value_sum", [warning["feature"] for warning in prediction["drift_warnings"]])
            invalid = {**features, "customer_state": None}
            self.assertEqual(client.post("/explain", json={"features": invalid}).status_code, 422)

    def test_packaged_model_accepts_swagger_example_and_rejects_categorical_nulls(self):
        with TestClient(api.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            metadata = client.get("/model-info").json()
            features = deepcopy(api.PredictionRequest.model_fields["features"].examples[0])
            examples = [deepcopy(features)]
            for name in metadata["features"]["categorical"]:
                features[name] = None
            self.assertEqual(client.post("/predict", json={"features": features}).status_code, 422)
            for example in examples:
                response = client.post("/predict", json={"features": example})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertTrue(0 <= result["late_probability"] <= 1)
                self.assertEqual(result["model_code"], "C3")
                self.assertEqual(result["is_predicted_late"], result["late_probability"] >= result["threshold"])


if __name__ == "__main__":
    unittest.main()
