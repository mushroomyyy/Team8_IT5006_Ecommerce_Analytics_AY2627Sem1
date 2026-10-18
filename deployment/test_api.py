"""Contract and input-validation tests independent of a trained artifact."""

from __future__ import annotations

import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from deployment import classification_api as api
from deployment.feature_validation import validate_features
from deployment.classification_service import DEFAULT_ARTIFACT


METADATA = {
    "model_code": "C3",
    "model_version": "test-model",
    "features": {
        "numeric": ["payment_value_sum", "order_approved_month", "order_approved_day_of_week", "order_item_count", "is_weekend_approval"],
        "categorical": ["customer_state"],
    },
}
FEATURES = {
    "payment_value_sum": 120.0,
    "order_approved_month": 6,
    "order_approved_day_of_week": 7,
    "order_item_count": 1,
    "is_weekend_approval": 0,
    "customer_state": "SP",
}


class FakeService:
    metadata = METADATA

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

    def test_explicit_missing_and_unseen_categories_are_supported(self):
        for features in ({**FEATURES, "payment_value_sum": None, "customer_state": None},
                         {**FEATURES, "customer_state": "unseen_state"}):
            self.assertEqual(self.client.post("/predict", json={"features": features}).status_code, 200)

    def test_all_numeric_missing_rejected(self):
        features = {**FEATURES, **dict.fromkeys(METADATA["features"]["numeric"])}
        self.assertEqual(self.client.post("/predict", json={"features": features}).status_code, 422)

    def test_nonfinite_numbers_rejected_before_inference(self):
        for value in (float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.assertRaisesRegex(ValueError, "finite number"):
                validate_features({**FEATURES, "payment_value_sum": value}, METADATA)

    def test_missing_values_are_normalized_for_fitted_imputer(self):
        frame = validate_features({**FEATURES, "payment_value_sum": None, "customer_state": None}, METADATA)
        self.assertTrue(np.isnan(frame.loc[0, "payment_value_sum"]))
        self.assertTrue(np.isnan(frame.loc[0, "customer_state"]))
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
    def test_packaged_model_accepts_swagger_example_and_all_categorical_nulls(self):
        with TestClient(api.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            metadata = client.get("/model-info").json()
            features = deepcopy(api.PredictionRequest.model_fields["features"].examples[0])
            examples = [deepcopy(features)]
            for name in metadata["features"]["categorical"]:
                features[name] = None
            examples.append(features)
            for example in examples:
                response = client.post("/predict", json={"features": example})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertTrue(0 <= result["late_probability"] <= 1)
                self.assertEqual(result["model_code"], "C3")
                self.assertEqual(result["is_predicted_late"], result["late_probability"] >= result["threshold"])


if __name__ == "__main__":
    unittest.main()
