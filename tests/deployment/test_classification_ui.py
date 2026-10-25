"""HTTP-only Streamlit workflow tests; no model deserialisation."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from deployment.app import orchestration
from deployment.app.orchestration import HEALTH_TIMEOUT, REQUEST_TIMEOUT, clear_caches
from deployment.app.app import BINARY, COUNTS, INTEGERS, LABELS, replay_score_percentiles, model_info, month_results, prepare_upload, scored_upload_frame, validate_form

ROOT = Path(__file__).resolve().parents[2] / "deployment"


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.ok = status < 400
        self.text = json.dumps(payload)

    def json(self): return self.payload


class ClassificationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats = json.loads((ROOT / 'artifacts/classification_model_stats.json').read_text())
        cls.samples = json.loads((ROOT / 'app/data/classification_sample_orders.json').read_text())

    def setUp(self):
        model_info.clear()
        clear_caches()
        replay_score_percentiles.clear()
        month_results.clear()
        self.demo_data = {'minimum_date': '2018-06-02', 'maximum_date': '2018-07-01', 'default_date': '2018-06-02', 'model_available_date': '2018-06-02', 'timezone': 'America/Sao_Paulo', 'batch_definition': 'Previous day by approval date', 'limitations': ['Snapshot history is incomplete.'], 'days': {}}
        orders = [{'order_id': r['order_id'], 'promised_delivery_date': '2018-06-20', 'features': {k: r[k] for k in self.stats['features']['numeric'] + self.stats['features']['categorical']}} for r in self.samples[:2]]
        from copy import deepcopy
        for day in ['2018-06-01', '2018-06-02', '2018-07-01']:  # 2 July is the default morning
            self.demo_data['days'][day] = deepcopy(orders)
        for row in self.demo_data['days']['2018-07-01']:
            row['promised_delivery_date'] = '2018-07-20'  # still ahead of the July cutoff
        self.json_reads = []
        def demo_json(name):
            self.json_reads.append(name)
            if name == 'simulation_orders.json': return self.demo_data
            if name == 'classification_sample_orders.json': return self.samples.copy()
            if name == 'simulation_outcomes.json': return {'outcomes': {self.samples[0]['order_id']: 1, self.samples[1]['order_id']: None}}
            if name == 'simulation_timing.json': return {'actual_days_from_promise': {self.samples[0]['order_id']: 1.0, self.samples[1]['order_id']: -6.0}}
            if name == 'replay_scores.json': return {'models': {'classification': self.stats['model_version'], 'regression': 'r1-test'}, 'generated_at_utc': 'test',
                                                   'scores': {r['order_id']: [.1 - .03 * i, -2.5] for i, r in enumerate(self.samples[:2])}}
            raise AssertionError(name)
        self.data_patch = patch('deployment.app.orchestration.read_demo_json', side_effect=demo_json)
        self.data_patch.start(); self.addCleanup(self.data_patch.stop)
        self.post_patch = patch('requests.post', side_effect=self.post_response)
        self.post = self.post_patch.start(); self.addCleanup(self.post_patch.stop)
        self.get_patch = patch('requests.get', side_effect=self.get_response)
        self.get = self.get_patch.start()
        self.addCleanup(self.get_patch.stop)
        self.app = AppTest.from_file(str(ROOT / 'app/app.py')).run(timeout=20)
        self.assertEqual(self.app.radio[0].value, 'Start here')  # usage guide is the landing page
        self.app.radio[0].set_value('Daily simulation').run(timeout=20)

    def result(self, probability=.064):
        return {'classification': {'late_probability': probability, 'model_code': 'C3', 'model_version': 'test', 'drift_warnings': []}, 'regression': {'predicted_days_from_promise': -2.5, 'model_code': 'R1'}}

    def get_response(self, url, **kwargs):
        if url.endswith('/health'): return Response({'status': 'healthy', 'model_loaded': True})
        if url.endswith('/model-info'): return Response({'classification': self.stats, 'regression': {'model_code': 'R1', 'model_version': 'r1-test'}})
        if url.endswith('/logs/summary'): return Response({'request_count': 3, 'prediction_count': 9, 'window_size': 9, 'mean_score': .08, 'recent_warnings': []})
        raise AssertionError(f'Frontend must not call demonstration API routes: {url}')

    def post_response(self, url, **kwargs):
        if url.endswith('/predict/batch'):
            return Response({'results': [{'index': i, 'order_id': row['order_id'], 'prediction': self.result(.1 - .03 * i)} for i, row in enumerate(kwargs['json']['rows'])]})
        if url.endswith('/predict'): return Response(self.result())
        raise AssertionError(url)

    def widget(self, collection, label):
        return next(w for w in getattr(self.app, collection) if w.label == label)

    def single(self):
        self.app.radio[0].set_value('Single-order API demo').run(timeout=20)

    def test_daily_default_and_navigation_boundaries(self):
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio[0].value, 'Daily simulation')
        self.assertEqual(str(self.app.date_input[0].value), '2018-07-02')  # first unseen (monitoring) morning
        self.widget('button', 'Next day').click().run(timeout=20)
        self.assertEqual(str(self.app.date_input[0].value), '2018-07-03')
        self.widget('button', 'Previous day').click().run(timeout=20)
        self.assertEqual(str(self.app.date_input[0].value), '2018-07-02')
        from datetime import date
        self.app.date_input[0].set_value(date(2018, 6, 2)).run(timeout=20)
        self.assertTrue(self.widget('button', 'Previous day').disabled)
        self.app.date_input[0].set_value(date(2018, 8, 30)).run(timeout=20)
        self.assertTrue(self.widget('button', 'Next day').disabled)
        self.assertFalse(self.app.exception)

    def test_outcomes_hidden_reset_on_date_and_policy_changes(self):
        self.assertNotIn('simulation_outcomes.json', self.json_reads)
        self.widget('button', 'Reveal eventual outcomes').click().run(timeout=20)
        self.assertTrue(any('Retrospective outcomes:' in w.value for w in self.app.warning))
        self.widget('slider', 'Follow-up proportion (%)').set_value(50).run(timeout=20)
        self.assertFalse(any('Retrospective outcomes:' in w.value for w in self.app.warning))
        self.widget('button', 'Reveal eventual outcomes').click().run(timeout=20)
        self.widget('button', 'Next day').click().run(timeout=20)
        self.assertFalse(any('Retrospective outcomes:' in w.value for w in self.app.warning))
        self.assertFalse(self.app.exception)

    def test_policy_and_unknown_denominators_without_rescoring_on_reveal(self):
        self.widget('slider', 'Follow-up proportion (%)').set_value(40).run(timeout=20)
        self.widget('number_input', 'Min. days before promise').set_value(3).run(timeout=20)
        calls = self.post.call_count
        self.widget('button', 'Reveal eventual outcomes').click().run(timeout=20)
        self.assertEqual(self.post.call_count, calls)
        self.assertIn('simulation_outcomes.json', self.json_reads)
        self.assertTrue(any('Unknown outcomes' in c.value for c in self.app.caption))
        table = next(d.value for d in self.app.dataframe if 'Population' in d.value.columns)
        self.assertEqual(list(table['Population']), ['all', 'eligible', 'selected'])
        self.assertEqual(list(table['unknown']), [1, 1, 0])

    def test_empty_day_and_failed_outcome_reveal(self):
        from datetime import date
        self.app.date_input[0].set_value(date(2018, 7, 1)).run(timeout=20)
        self.assertTrue(any('No eligible orders' in i.value for i in self.app.info))
        self.assertFalse(self.app.exception)
        with patch('deployment.app.orchestration.read_demo_json', side_effect=ValueError('outcomes unavailable')):
            self.widget('button', 'Reveal eventual outcomes').click().run(timeout=20)
        self.assertTrue(any('outcomes unavailable' in e.value for e in self.app.error))
        self.assertFalse(self.app.exception)

    def test_open_worklist_order(self):
        selected = self.widget('selectbox', 'Inspect a ranked order').value
        self.widget('button', 'Open selected order in API demo').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio[0].value, 'Single-order API demo')
        self.assertEqual(self.widget('selectbox', 'Held-out historical order').value, selected)

    def test_trained_dropdowns_integer_inputs_and_integrated_prediction(self):
        self.single()
        for name in self.stats['features']['categorical']:
            self.assertEqual(self.widget('selectbox', LABELS[name][0]).options, self.stats['categories'][name])
        for name in COUNTS:
            field = self.widget('number_input', LABELS[name][0])
            self.assertIsInstance(field.value, int)
            self.assertEqual(field.step, 1)
        with patch('requests.post', return_value=Response(self.result())) as post:
            self.widget('button', 'Predict delivery').click().run(timeout=20)
        features = post.call_args.kwargs['json']['features']
        self.assertEqual(set(features), set(self.stats['features']['numeric'] + self.stats['features']['categorical']))
        for name in INTEGERS | BINARY: self.assertIsInstance(features[name], int)
        self.assertEqual(self.app.metric[0].value, '6.4%')
        self.assertEqual(self.app.metric[1].value, '-2.50 days')
        self.assertFalse(self.app.exception)

    def test_sample_switch_resets_edited_input(self):
        self.single()
        original = self.widget('number_input', 'Payment value (R$)').value
        self.widget('number_input', 'Payment value (R$)').set_value(original + 50).run(timeout=20)
        picker = self.widget('selectbox', 'Held-out historical order')
        first, second = picker.options[:2]
        picker.select(second).run(timeout=20)
        self.widget('selectbox', 'Held-out historical order').select(first).run(timeout=20)
        self.assertEqual(self.widget('number_input', 'Payment value (R$)').value, original)

    def test_invalid_calendar_and_invalid_json_block_submission(self):
        self.single()
        self.widget('number_input', 'Approval month').set_value(2)
        self.widget('number_input', 'Approval day of month').set_value(31).run(timeout=20)
        self.assertTrue(self.widget('button', 'Predict delivery').disabled)
        self.widget('checkbox', 'Use raw JSON instead of the form').check()
        self.app.text_area[0].set_value('{').run(timeout=20)
        self.assertTrue(self.widget('button', 'Predict delivery').disabled)
        self.assertFalse(self.app.exception)

    def test_numeric_null_and_range_warning(self):
        self.single()
        self.widget('number_input', 'Payment value (R$)').set_value(1e10).run(timeout=20)
        self.assertTrue(any('outside refit range' in w.value for w in self.app.warning))
        sample_id = self.widget('selectbox', 'Held-out historical order').value
        next(c for c in self.app.checkbox if c.key == f'input_{sample_id}_payment_value_sum_missing').check().run(timeout=20)
        with patch('requests.post', return_value=Response(self.result())) as post:
            self.widget('button', 'Predict delivery').click().run(timeout=20)
        self.assertIsNone(post.call_args.kwargs['json']['features']['payment_value_sum'])

    def test_http_validation_connection_and_timeout_errors(self):
        self.single()
        with patch('requests.post', return_value=Response({'detail': 'Invalid payment'}, 422)):
            self.widget('button', 'Predict delivery').click().run(timeout=20)
        self.assertTrue(any('422: Invalid payment' in e.value for e in self.app.error))
        with patch('requests.post', side_effect=requests.Timeout('Timed out')):
            self.widget('button', 'Predict delivery').click().run(timeout=20)
        self.assertTrue(any('Timed out' in e.value for e in self.app.error))
        self.get.side_effect = requests.ConnectionError('backend unavailable')
        with patch.object(orchestration, 'WAKE_BUDGET_SECONDS', 0):
            self.widget('button', 'Retry API connection').click().run(timeout=20)
        self.assertTrue(any('backend unavailable' in e.value for e in self.app.error))
        self.assertFalse(self.app.session_state['api_connected'])
        self.assertFalse(self.app.exception)

    def test_cold_start_note_timeout_budget_and_retry_after_inactivity(self):
        self.assertTrue(any('50 seconds or more' in note.value and 'After inactivity' in note.value for note in self.app.caption))
        self.assertTrue(self.app.session_state['api_connected'])
        from streamlit.delta_generator import DeltaGenerator
        with patch.object(DeltaGenerator, 'status', autospec=True, side_effect=DeltaGenerator.status) as status:
            self.widget('button', 'Retry API connection').click().run(timeout=20)
        self.assertTrue(any('Checking / waking' in call.args[1] for call in status.call_args_list))
        # No single call may block a run (or a refreshed page) for minutes.
        for call in self.get.call_args_list: self.assertEqual(call.kwargs['timeout'], HEALTH_TIMEOUT)
        for call in self.post.call_args_list: self.assertEqual(call.kwargs['timeout'], REQUEST_TIMEOUT)
        self.assertLessEqual(sum(REQUEST_TIMEOUT), 65)
        self.get.side_effect = requests.Timeout('Service still waking')
        with patch.object(orchestration, 'WAKE_BUDGET_SECONDS', 0):
            self.widget('button', 'Retry API connection').click().run(timeout=20)
        self.assertFalse(self.app.session_state['api_connected'])
        self.assertTrue(any('Retry API connection' in note.value and '50 seconds or more' in note.value for note in self.app.info))
        self.assertFalse(self.app.exception)
        self.get.side_effect = self.get_response
        self.widget('button', 'Retry API connection').click().run(timeout=20)
        self.assertTrue(self.app.session_state['api_connected'])
        self.assertFalse(self.app.exception)

    def test_model_or_service_changes_invalidate_worklist_cache(self):
        first_calls = self.post.call_count
        self.app.run(timeout=20)
        self.assertEqual(self.post.call_count, first_calls)
        self.stats = {**self.stats, 'model_version': 'updated-model'}
        self.widget('button', 'Retry API connection').click().run(timeout=20)
        self.assertGreater(self.post.call_count, first_calls)
        self.assertNotIn('simulation_outcomes.json', self.json_reads)
        current_calls = self.post.call_count
        with patch.dict(os.environ, {'DELIVERY_API_URL': 'https://another-inference-api.example'}):
            self.app.run(timeout=20)
        self.assertGreater(self.post.call_count, current_calls)

    def test_all_missing_and_huge_values_validate_without_crash(self):
        features = {k: self.samples[0][k] for k in self.stats['features']['numeric'] + self.stats['features']['categorical']}
        features['payment_value_sum'] = 10 ** 500
        self.assertTrue(validate_form(features, self.stats))
        for key in self.stats['features']['numeric']: features[key] = None
        self.assertTrue(any('at least one numeric' in e for e in validate_form(features, self.stats)))

    def test_reupload_handles_null_and_ignores_prior_output(self):
        cols = self.stats['features']['numeric'] + self.stats['features']['categorical']
        frame = pd.DataFrame([{**{k: self.samples[0][k] for k in cols}, 'order_id': 'abc', 'late_probability': .1, 'rank': 1}])
        frame.loc[0, 'payment_value_sum'] = None
        kept, rows, error = prepare_upload(frame, cols)
        self.assertIsNone(error)
        self.assertEqual(list(kept), ['order_id'] + cols)
        self.assertIsNone(rows[0]['features']['payment_value_sum'])
        self.assertEqual(rows[0]['order_id'], 'abc')
        json.dumps(rows)
        _, _, error = prepare_upload(frame.drop(columns=[cols[0]]).assign(surprise=1), cols)
        self.assertIn('Missing feature columns:', error)
        self.assertIn('Unrecognised columns: surprise', error)

    def test_scored_csv_round_trip_with_complete_prediction_contract(self):
        # Mirrors format_prediction fields from both HTTP services without importing models.
        cols = self.stats['features']['numeric'] + self.stats['features']['categorical']
        incoming = pd.DataFrame([{**{key: self.samples[0][key] for key in cols}, 'order_id': 'round-trip'}])
        incoming.loc[0, 'payment_value_sum'] = None
        prediction = {'classification': {
            'late_probability': .064, 'drift_warnings': [],
            'score_interpretation': 'Estimated late-delivery risk; not a calibrated guarantee.',
            'model_code': 'C3', 'model_version': 'fixture'}, 'regression': {
            'predicted_days_from_promise': -2.5, 'raw_days_from_promise': -2.5,
            'late_by_days': 0., 'early_by_days': 2.5, 'clipping_applied': True,
            'clipping_policy': 'Predicted approval-to-delivery duration bounded to 0–45 days, matching Phase 2 evaluation.',
            'target': 'days from promised delivery date; negative means early, positive means late',
            'model_code': 'R1', 'model_version': 'fixture'}}
        scored = scored_upload_frame(incoming, {'results': [{'index': 0, 'order_id': 'round-trip', 'prediction': prediction}]})
        import io
        uploaded = pd.read_csv(io.StringIO(scored.to_csv(index=False)), float_precision='round_trip')
        kept, rows, error = prepare_upload(uploaded, cols)
        self.assertIsNone(error)
        self.assertEqual(list(kept), ['order_id'] + cols)
        self.assertEqual(rows[0]['order_id'], 'round-trip')
        self.assertIsNone(rows[0]['features']['payment_value_sum'])
        for key in cols:
            if key != 'payment_value_sum':
                self.assertEqual(rows[0]['features'][key], self.samples[0][key])

    def test_policy_changes_rerank_locally_without_api_calls(self):
        calls = self.post.call_count
        self.widget('slider', 'Follow-up proportion (%)').set_value(50).run(timeout=20)
        self.widget('number_input', 'Min. days before promise').set_value(2).run(timeout=20)
        self.assertEqual(self.post.call_count, calls)
        self.assertEqual(self.app.metric[3].value, '1')
        self.assertFalse(self.app.exception)

    def test_day_navigation_is_clamped_instead_of_crashing(self):
        from datetime import date
        from deployment.app import app as app_module
        low, high = date(2018, 6, 2), date(2018, 8, 30)
        # Queued clicks can run the callback after the button should have been disabled.
        for start, delta, expected in [(high, 1, high), (low, -1, low), (date(2018, 6, 10), 1, date(2018, 6, 11))]:
            with patch.object(app_module.st, 'session_state', {'simulation_day': start}) as state:
                app_module.navigate_day(delta, low, high)
                self.assertEqual(state['simulation_day'], expected)
        self.app.session_state['simulation_day'] = date(2018, 9, 9)  # stale out-of-range state must not crash the page
        self.app.run(timeout=20)
        self.assertFalse(self.app.exception)

    def test_new_session_reuses_scores_and_connection_is_checked_once(self):
        health_calls = sum(call.args[0].endswith('/health') for call in self.get.call_args_list)
        self.app.run(timeout=20)
        self.assertEqual(sum(call.args[0].endswith('/health') for call in self.get.call_args_list), health_calls)
        posts = self.post.call_count
        viewer = AppTest.from_file(str(ROOT / 'app/app.py')).run(timeout=20)  # e.g. a refresh or a second viewer
        viewer.radio[0].set_value('Daily simulation').run(timeout=20)
        self.assertEqual(self.post.call_count, posts)

    def test_month_end_review_works_without_api(self):
        posts = self.post.call_count
        self.get.side_effect = requests.ConnectionError('backend asleep')
        self.app.radio[0].set_value('Month-end review').run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertEqual(self.post.call_count, posts)
        labels = [m.label for m in self.app.metric]
        self.assertIn('Late deliveries caught', labels)
        caught = next(m for m in self.app.metric if m.label == 'Late deliveries caught')
        self.assertEqual(caught.value, '1 of 1')  # opens on July: one fixture day, its late order ranks first
        self.assertTrue(any('Monitoring month' in i.value and 'Retrospective view' in i.value for i in self.app.info))
        mae = next(m for m in self.app.metric if m.label == 'Typical error (MAE)')
        self.assertEqual(mae.value, '3.5 days')  # |-2.5 - 1| and |-2.5 + 6|
        self.app.radio(key='month_pick').set_value('2018-06').run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Selection month' in w.value for w in self.app.warning))
        self.assertEqual(next(m for m in self.app.metric if m.label == 'Late deliveries caught').value, '2 of 2')

    def test_example_batch_csv_matches_upload_template(self):
        cols = self.stats['features']['numeric'] + self.stats['features']['categorical']
        example = pd.read_csv(ROOT.parent / 'docs/examples/batch_example.csv', float_precision='round_trip')
        self.assertEqual(list(example.columns), ['order_id'] + cols)
        kept, rows, error = prepare_upload(example, cols)
        self.assertIsNone(error)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertFalse(validate_form(row['features'], self.stats))  # every example order passes the app's checks

    def test_start_page_explains_every_page_without_api(self):
        self.get.side_effect = requests.ConnectionError('backend asleep')
        self.app.radio[0].set_value('Start here').run(timeout=20)
        self.assertFalse(self.app.exception)
        text = ' '.join(m.value for m in self.app.markdown)
        for page in ['Daily simulation', 'Month-end review', 'Explore the data', 'Single-order API demo', 'Batch API demo', 'Models and guidance']:
            self.assertIn(page, text)

    def test_explore_page_uses_packaged_tables_and_filters(self):
        posts = self.post.call_count
        self.app.radio[0].set_value('Explore the data').run(timeout=60)
        self.assertFalse(self.app.exception)
        self.assertEqual(self.post.call_count, posts)
        total = int(next(m for m in self.app.metric if m.label == 'Approved orders').value.replace(',', ''))
        self.assertGreater(total, 90000)
        self.app.multiselect(key='eda_states').set_value(['SP']).run(timeout=60)
        self.assertFalse(self.app.exception)
        filtered = int(next(m for m in self.app.metric if m.label == 'Approved orders').value.replace(',', ''))
        self.assertLess(filtered, total)

    def test_supporting_pages_and_frontend_isolation(self):
        for page in ['Batch API demo', 'Models and guidance', 'Month-end review', 'Start here']:
            self.app.sidebar.radio[0].set_value(page).run(timeout=20)
            self.assertFalse(self.app.exception)
        source = (ROOT / 'app/app.py').read_text()
        for text in ['import joblib', 'import sklearn', 'classification_service', 'top_10_percent_boundary', 'is_predicted_late']:
            # Legacy CSV result columns are intentionally accepted during re-upload.
            if text == 'is_predicted_late': continue
            self.assertNotIn(text, source)


if __name__ == '__main__': unittest.main()
