"""Exercise Streamlit reruns and HTTP errors against real artifact metadata."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from deployment.classification_service import ARTIFACTS, ClassificationService


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.ok = status < 400
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class ClassificationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ClassificationService()
        cls.samples = json.loads((ARTIFACTS / "classification_sample_orders.json").read_text())

    def setUp(self):
        self.get_patch = patch("requests.get", return_value=Response(self.service.metadata))
        self.get = self.get_patch.start()
        self.addCleanup(self.get_patch.stop)
        path = Path(__file__).with_name("classification_streamlit.py")
        self.app = AppTest.from_file(str(path)).run()

    def test_render_and_sample_switch_reset_features(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.metric), 3)
        self.assertEqual(self.app.metric[0].value, "11.22%")
        self.app.text_area[0].set_value('{"edited": true}').run()
        self.app.selectbox[0].select(self.samples[1]["order_id"]).run()
        self.assertEqual(len(self.app.exception), 0)
        actual = json.loads(self.app.text_area[0].value)
        self.assertEqual(actual, {k: v for k, v in self.samples[1].items() if k != "order_id"})

    def test_prediction_submits_current_features_to_http_api(self):
        features = json.loads(self.app.text_area[0].value)
        result = self.service.predict(features)
        with patch("requests.post", return_value=Response(result)) as post:
            self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(post.call_args.kwargs["json"], {"features": features})
        self.assertTrue(post.call_args.args[0].endswith("/predict"))
        self.assertEqual(self.app.metric[0].value, f"{result['late_probability']:.1%}")

    def test_validation_and_connection_errors_are_visible(self):
        with patch("requests.post", return_value=Response({"detail": "Invalid payment value"}, 422)):
            self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("422: Invalid payment value", self.app.error[0].value)
        self.get.side_effect = requests.ConnectionError("backend unavailable")
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("backend unavailable", self.app.error[0].value)

    def test_malformed_json_is_rejected_before_http_request(self):
        self.app.text_area[0].set_value("{")
        with patch("requests.post") as post:
            self.app.button[0].click().run()
        post.assert_not_called()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)


if __name__ == "__main__":
    unittest.main()
