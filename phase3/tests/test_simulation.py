"""Policy boundary checks that would catch future-information leakage."""
from copy import deepcopy
from unittest.mock import patch
from phase3.frontend.orchestration import score_demo_rows
from datetime import date
import unittest

from phase3.frontend.simulation import build_worklist, retrospective, SimulationDataError
import json


def score(rows):
    return [{"classification": {"late_probability": row["features"].get("score", .1)},
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
        def with_invalid(rows):
            return [None if row['order_id'] == 'c' else score([row])[0] for row in rows]
        result = build_worklist(self.data, date(2018, 6, 2), .3, 1, with_invalid)
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



class HttpOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'order_id': 'a', 'features': {'score': .1}}, {'order_id': 'b', 'features': {'score': .2}}]
        self.results = [{'index': i, 'order_id': row['order_id'], 'prediction': score([row])[0]} for i, row in enumerate(self.rows)]

    def test_validation422_only_becomes_exclusion_and_remote_errors_abort(self):
        results = deepcopy(self.results)
        results[0] = {'index': 0, 'order_id': 'a', 'status_code': 422, 'error': 'invalid features'}
        with patch('phase3.frontend.orchestration.api_request', return_value={'results': results}):
            self.assertEqual(score_demo_rows('https://api', self.rows), [None, self.results[1]['prediction']])
        for status in [500, 503, None]:
            results[0]['status_code'] = status
            with self.subTest(status=status), patch('phase3.frontend.orchestration.api_request', return_value={'results': results}), self.assertRaises(SimulationDataError):
                score_demo_rows('https://api', self.rows)

    def test_results_require_exact_count_index_identity_and_finite_outputs(self):
        broken = []
        broken.append(self.results[:1])
        broken.append(list(reversed(self.results)))
        for change in ({'order_id': 'other'}, {'index': 2}, {'prediction': {'classification': {'late_probability': float('nan')}, 'regression': {'predicted_days_from_promise': 1}}}):
            entries = deepcopy(self.results)
            entries[0].update(change)
            broken.append(entries)
        for results in broken:
            with self.subTest(results=results), patch('phase3.frontend.orchestration.api_request', return_value={'results': results}), self.assertRaises(SimulationDataError):
                score_demo_rows('https://api', self.rows)

    def test_preeligibility_sends_only_eligible_features_and_batch_identity(self):
        data = {'days': {'2018-06-01': [order('ok'), order('due', promise='2018-06-02'), order('past', delivery='2018-06-01T23:59:59')]}, 'timezone': 'local', 'batch_definition': 'approval_date'}
        def response(url, endpoint, payload):
            self.assertEqual(endpoint, '/predict/batch')
            self.assertEqual([r['order_id'] for r in payload['rows']], ['ok'])
            return {'results': [{'index': 0, 'order_id': 'ok', 'prediction': score(payload['rows'])[0]}]}
        with patch('phase3.frontend.orchestration.api_request', side_effect=response):
            result = build_worklist(data, date(2018, 6, 2), .3, 1, lambda rows: score_demo_rows('https://api', rows))
        self.assertEqual(result['counts'], {'total': 3, 'eligible': 1, 'selected': 1, 'excluded': 2})

    def test_invalid_policy_rejected_before_scoring(self):
        data = {'days': {}, 'timezone': 'local', 'batch_definition': 'approval_date'}
        for proportion, remaining in [(-1, 1), (1.1, 1), (float('nan'), 1), (.3, -1), (.3, 1.5), (.3, True)]:
            with self.subTest(proportion=proportion, remaining=remaining), self.assertRaises(ValueError):
                build_worklist(data, date(2018, 6, 2), proportion, remaining, lambda _: self.fail('should not score'))


if __name__ == '__main__': unittest.main()
