"""Policy boundary checks that would catch future-information leakage."""
from copy import deepcopy
from datetime import date
import unittest

from fastapi.testclient import TestClient
from phase3.backend import api
from phase3.backend.simulation import build_worklist, retrospective, SimulationDataError
from phase3.backend.feature_validation import FeatureValidationError
from phase3.backend.model_service import ModelService
from phase3.backend.classification_service import ClassificationService, ARTIFACTS
import json
import numpy as np


def score(rows):
    return [{"classification": {"late_probability": row.get("score", .1)},
             "regression": {"predicted_days_from_promise": -2.}} for row in rows]


def order(key, delivery=None, promise="2018-06-10", risk=.1):
    return {"order_id": key, "approved_at": "2018-06-01T12:00:00", "promised_delivery_date": promise,
            "delivered_at": delivery, "features": {"score": risk}}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.data = {"timezone": "America/Sao_Paulo (naive source dates)", "batch_definition": "approval_date",
                     "days": {"2018-06-01": [order("b"), order("a"), order("c", risk=.3),
                       order("past", "2018-06-01T23:59:59"), order("at", "2018-06-02T00:00:00"),
                       order("future", "2018-06-02T00:00:01"), order("due", promise="2018-06-03"),
                       order("missing", promise=None)]}}

    def run_policy(self, proportion=.3, data=None, day=date(2018, 6, 2)):
        return build_worklist(data or self.data, day, proportion, 1, score)

    def test_cutoff_exact_boundary_ties_ceiling_and_counts(self):
        result = self.run_policy()
        self.assertEqual(result["counts"], {"total": 8, "eligible": 5, "excluded": 3, "selected": 2})
        self.assertEqual([row["order_id"] for row in result["orders"]], ["c", "a", "at", "b", "future"])
        self.assertEqual([row["order_id"] for row in result["orders"] if row["follow_up_recommended"]], ["c", "a"])
        self.assertEqual(result["exclusion_reasons"], {"already_delivered": 1, "insufficient_time_remaining": 1,
                                                      "missing_promised_delivery_date": 1})
        self.assertNotIn("delivered_at", str(result))
        self.assertNotIn("outcome", str(result))

    def test_future_delivery_does_not_change_score_or_eligibility(self):
        changed = deepcopy(self.data)
        changed["days"]["2018-06-01"][5]["delivered_at"] = "2018-12-31T12:00:00"
        self.assertEqual(self.run_policy(), self.run_policy(data=changed))
        # A final cancelled status has no cancellation time and cannot remove a row.
        changed["days"]["2018-06-01"][5]["order_status"] = "canceled"
        self.assertEqual(self.run_policy(), self.run_policy(data=changed))

    def test_proportions_empty_days_and_supported_dates(self):
        self.assertEqual(self.run_policy(0)["counts"]["selected"], 0)
        self.assertEqual(self.run_policy(1)["counts"]["selected"], 5)
        self.assertEqual(self.run_policy(day=date(2018, 7, 1))["counts"]["total"], 0)
        for day in (date(2018, 6, 1), date(2018, 7, 2)):
            with self.assertRaises(ValueError):
                self.run_policy(day=day)

    def test_invalid_feature_exclusion_model_availability_and_scorer_count(self):
        def validate(row):
            if row["score"] == .3:
                raise FeatureValidationError("bad feature")
        result = build_worklist(self.data, date(2018, 6, 2), .3, 1, score, validate)
        self.assertEqual(result["counts"]["eligible"], 4)
        self.assertEqual(result["exclusion_reasons"]["invalid_model_inputs"], 1)
        self.assertNotIn("c", [row["order_id"] for row in result["orders"]])
        data = {**self.data, "model_available_date": "2018-06-03"}
        with self.assertRaisesRegex(ValueError, "2018-06-03"):
            self.run_policy(data=data)
        with self.assertRaises(SimulationDataError):
            build_worklist(self.data, date(2018, 6, 2), .3, 1, lambda rows: [])
        malformed = deepcopy(self.data)
        malformed["days"]["2018-06-01"][0]["delivered_at"] = "not-a-date"
        with self.assertRaises(SimulationDataError):
            self.run_policy(data=malformed)

    def test_retrospective_metrics_denominators_and_unknowns(self):
        policy = self.run_policy()
        revealed = retrospective(policy, {"c": 1, "a": None, "past": 1, "at": 0, "b": 0, "future": 0})
        m = revealed["metrics"]
        self.assertEqual(m["all"], {"total": 8, "known": 5, "unknown": 3, "late": 2})
        self.assertEqual(m["eligible"], {"total": 5, "known": 4, "unknown": 1, "late": 1})
        self.assertEqual(m["selected"], {"total": 2, "known": 1, "unknown": 1, "late": 1})
        self.assertEqual(m["selected_late_rate"], 1)
        self.assertEqual(m["eligible_late_coverage"], 1)
        self.assertEqual(m["all_late_coverage"], .5)
        empty = retrospective(self.run_policy(0), {})["metrics"]
        self.assertIsNone(empty["selected_late_rate"])
        self.assertIsNone(empty["eligible_late_coverage"])


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ModelService()
        cls.features = json.loads((ARTIFACTS / "classification_sample_orders.json").read_text())[0]
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


class DailyApiTests(unittest.TestCase):
    def test_policy_parameter_validation_and_outcomes_separation(self):
        with TestClient(api.app) as client:
            for params in ({"simulation_date": "2018-06-01"}, {"simulation_date": "2018-07-02"},
                           {"review_proportion": -1}, {"review_proportion": 1.1},
                           {"minimum_days_remaining": -1}, {"minimum_days_remaining": 1.5}, {"simulation_date": "bad"}):
                self.assertEqual(client.get("/worklist", params=params).status_code, 422)
            config = client.get("/simulation").json()
            self.assertEqual(config["minimum_date"], "2018-06-02")
            response = client.get("/worklist")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("delivered_at", response.text)
            self.assertNotIn('"outcomes"', response.text)
            reveal = client.get("/outcomes")
            self.assertEqual(reveal.status_code, 200, reveal.text)
            self.assertTrue(reveal.json()["display_only"])
            self.assertEqual(reveal.json()["metrics"]["all"]["total"], response.json()["counts"]["total"])
            self.assertEqual(client.get("/worklist").json(), response.json())


if __name__ == "__main__":
    unittest.main()
