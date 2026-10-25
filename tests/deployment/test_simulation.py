"""Policy boundary checks that would catch future-information leakage."""
from copy import deepcopy
from unittest.mock import patch
from deployment.app import orchestration
from deployment.app.orchestration import score_demo_rows, wait_for_service
from datetime import date
import unittest

from deployment.app.simulation import build_worklist, month_replay, month_summary, monthly_ranking, retrospective, SimulationDataError
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
        self.assertEqual(self.run_policy(day=date(2018, 8, 30))["counts"]["total"], 0)
        for day in (date(2018, 6, 1), date(2018, 8, 31)):
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

    def test_month_replay_matches_daily_worklists_and_counts_unknowns(self):
        scores = {row["order_id"]: [row["features"]["score"], -1.] for row in self.data["days"]["2018-06-01"]}
        outcomes = {"c": 1, "a": None, "past": 1, "at": 0, "b": 1, "future": 0}
        replay = month_replay(self.data, scores, outcomes, 1, "2018-06")
        self.assertEqual(len(replay), 30)
        self.assertEqual(len(month_replay(self.data, scores, outcomes, 1)), 90)  # 2 Jun to 30 Aug
        first = month_summary(replay, .3)["days"][0]
        expected = retrospective(self.run_policy(), outcomes)["metrics"]
        self.assertEqual((first["eligible"], first["selected"], first["late_caught"], first["late_all"]),
                         (expected["eligible"]["total"], expected["selected"]["total"], expected["selected"]["late"], expected["all"]["late"]))
        pooled = month_summary(replay, .3)["pooled"]
        self.assertEqual(pooled["selected_known"], 1)  # "a" is selected but unknown: excluded from the hit-rate denominator
        self.assertEqual(month_summary(replay, 0)["pooled"]["all_late_coverage"], 0)
        self.assertIsNone(month_summary(month_replay(self.data, scores, {}, 1, "2018-06"), 1)["pooled"]["all_late_coverage"])
        with self.assertRaises(SimulationDataError):
            month_replay(self.data, {}, outcomes, 1)

    def test_timing_accuracy_matches_hand_calculation_and_month_filter(self):
        from deployment.app.simulation import timing_accuracy
        data = {"days": {"2018-06-01": [{"order_id": "x"}, {"order_id": "y"}], "2018-07-01": [{"order_id": "z"}], "2018-06-02": [{"order_id": "gone"}]}}
        scores = {"x": [.9, -10.], "y": [.1, -4.], "z": [.5, 0.], "gone": [.2, -1.]}
        actual = {"x": -12., "y": 2., "z": 3., "gone": None}  # canceled/unknown targets are skipped
        june = timing_accuracy(data, scores, actual, {"x": 0, "y": 1}, "2018-06")
        self.assertEqual(june["n"], 2)
        self.assertAlmostEqual(june["mae"], 4.0)            # |2| and |-6|
        self.assertAlmostEqual(june["rmse"], 20 ** 0.5)      # sqrt((4 + 36) / 2)
        self.assertAlmostEqual(june["bias"], -2.0)
        self.assertEqual((june["predicted_late_orders"], june["predicted_on_time_orders"]), (-4., -10.))
        self.assertEqual([d["approval_date"] for d in june["days"]], ["2018-06-01"])
        self.assertEqual(timing_accuracy(data, scores, actual, {})["n"], 3)
        self.assertIsNone(timing_accuracy(data, scores, {}, {}))

    def test_monthly_ranking_uses_whole_cohort(self):
        scores = {"x": [.9, 0.], "y": [.5, 0.], "z": [.1, 0.]}
        data = {"days": {"2018-06-01": [{"order_id": "x"}], "2018-06-02": [{"order_id": "y"}, {"order_id": "z"}]}}
        result = monthly_ranking(data, scores, {"x": 0, "y": 1, "z": 1}, .5)
        self.assertEqual((result["selected"], result["late_caught"], result["late_all"]), (2, 1, 2))
        self.assertEqual(monthly_ranking(data, scores, {"x": 0, "y": 1, "z": 1}, 1, "2018-06")["total"], 3)
        self.assertEqual(monthly_ranking(data, scores, {"x": 0, "y": 1, "z": 1}, 1, "2018-07")["total"], 0)



class HttpOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'order_id': 'a', 'features': {'score': .1}}, {'order_id': 'b', 'features': {'score': .2}}]
        self.results = [{'index': i, 'order_id': row['order_id'], 'prediction': score([row])[0]} for i, row in enumerate(self.rows)]

    def test_host_html_error_pages_become_readable_messages(self):
        from deployment.app.orchestration import api_request
        class Reply:
            def __init__(self, status, text, payload=None): self.status_code, self.ok, self.text, self.payload = status, status < 400, text, payload
            def json(self):
                if self.payload is None: raise ValueError('not json')
                return self.payload
        with patch('requests.get', return_value=Reply(502, '<!DOCTYPE html><html><body>Bad gateway</body></html>')):
            with self.assertRaises(ValueError) as caught:
                api_request('https://api', '/health')
        self.assertIn('502', str(caught.exception))
        self.assertIn('restarting', str(caught.exception))
        self.assertNotIn('<html', str(caught.exception))
        with patch('requests.post', return_value=Reply(422, '{}', {'detail': 'customer_state must exactly match one of: SP'})):
            with self.assertRaises(ValueError) as caught:
                api_request('https://api', '/predict', {'features': {}})
        self.assertIn('customer_state must exactly match', str(caught.exception))  # our API's own messages pass through

    def test_validation422_only_becomes_exclusion_and_remote_errors_abort(self):
        results = deepcopy(self.results)
        results[0] = {'index': 0, 'order_id': 'a', 'status_code': 422, 'error': 'invalid features'}
        with patch('deployment.app.orchestration.api_request', return_value={'results': results}):
            self.assertEqual(score_demo_rows('https://api', self.rows), [None, self.results[1]['prediction']])
        for status in [500, 503, None]:
            results[0]['status_code'] = status
            with self.subTest(status=status), patch('deployment.app.orchestration.api_request', return_value={'results': results}), self.assertRaises(SimulationDataError):
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
            with self.subTest(results=results), patch('deployment.app.orchestration.api_request', return_value={'results': results}), self.assertRaises(SimulationDataError):
                score_demo_rows('https://api', self.rows)

    def test_preeligibility_sends_only_eligible_features_and_batch_identity(self):
        data = {'days': {'2018-06-01': [order('ok'), order('due', promise='2018-06-02'), order('past', delivery='2018-06-01T23:59:59')]}, 'timezone': 'local', 'batch_definition': 'approval_date'}
        def response(url, endpoint, payload):
            self.assertEqual(endpoint, '/predict/batch')
            self.assertEqual([r['order_id'] for r in payload['rows']], ['ok'])
            return {'results': [{'index': 0, 'order_id': 'ok', 'prediction': score(payload['rows'])[0]}]}
        with patch('deployment.app.orchestration.api_request', side_effect=response):
            result = build_worklist(data, date(2018, 6, 2), .3, 1, lambda rows: score_demo_rows('https://api', rows))
        self.assertEqual(result['counts'], {'total': 3, 'eligible': 1, 'selected': 1, 'excluded': 2})

    def test_scoring_candidates_cover_every_minimum_days_setting_without_delivered_rows(self):
        from deployment.app.simulation import scoring_candidates
        data = {'days': {'2018-06-01': [order('ok'), order('due', promise='2018-06-02'), order('missing', promise=None),
                                        order('past', delivery='2018-06-01T23:59:59'), order('later', delivery='2018-06-05T00:00:00')]},
                'timezone': 'local', 'batch_definition': 'approval_date'}
        self.assertEqual([r['order_id'] for r in scoring_candidates(data, date(2018, 6, 2))], ['ok', 'due', 'later'])
        for remaining in (0, 1, 7):
            eligible = build_worklist(data, date(2018, 6, 2), 1, remaining, score)['orders']
            self.assertLessEqual({r['order_id'] for r in eligible}, {'ok', 'due', 'later'})
        with self.assertRaises(ValueError):
            scoring_candidates(data, date(2018, 8, 31))

    def test_batches_are_chunked_with_progress(self):
        rows = [{'order_id': str(i), 'features': {'score': .1}} for i in range(250)]
        def response(url, endpoint, payload):
            return {'results': [{'index': i, 'order_id': r['order_id'], 'prediction': score([r])[0]} for i, r in enumerate(payload['rows'])]}
        progress = []
        with patch('deployment.app.orchestration.api_request', side_effect=response) as call:
            self.assertEqual(len(score_demo_rows('https://api', rows, lambda done, total: progress.append((done, total)))), 250)
        self.assertEqual([len(c.args[2]['rows']) for c in call.call_args_list], [100, 100, 50])
        self.assertEqual(progress, [(0, 250), (100, 250), (200, 250), (250, 250)])

    def test_wait_for_service_polls_briefly_then_gives_up(self):
        import requests
        attempts, waits = iter([requests.ConnectionError('asleep'), requests.Timeout('slow'), {'status': 'ok'}]), []
        def health(*args, **kwargs):
            item = next(attempts)
            if isinstance(item, Exception): raise item
            return item
        with patch('deployment.app.orchestration.api_request', side_effect=health), patch('time.sleep') as sleep:
            self.assertEqual(wait_for_service('https://api', lambda elapsed, exc: waits.append(type(exc).__name__)), {'status': 'ok'})
        self.assertEqual(waits, ['ConnectionError', 'Timeout'])
        self.assertEqual(sleep.call_count, 2)
        with patch('deployment.app.orchestration.api_request', side_effect=requests.ConnectionError('down')), \
             patch.object(orchestration, 'WAKE_BUDGET_SECONDS', 0), self.assertRaises(requests.Timeout):
            wait_for_service('https://api')

    def test_invalid_policy_rejected_before_scoring(self):
        data = {'days': {}, 'timezone': 'local', 'batch_definition': 'approval_date'}
        for proportion, remaining in [(-1, 1), (1.1, 1), (float('nan'), 1), (.3, -1), (.3, 1.5), (.3, True)]:
            with self.subTest(proportion=proportion, remaining=remaining), self.assertRaises(ValueError):
                build_worklist(data, date(2018, 6, 2), proportion, remaining, lambda _: self.fail('should not score'))


if __name__ == '__main__': unittest.main()
