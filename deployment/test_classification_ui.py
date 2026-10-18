"""HTTP-only frontend tests: metadata fixtures never deserialize the fitted model."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

ARTIFACTS = Path(__file__).with_name('artifacts')


class Response:
    def __init__(self,payload,status=200):
        self.payload,self.status_code=payload,status
        self.ok=status<400
        self.text=json.dumps(payload)
    def json(self):return self.payload


class ClassificationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats=json.loads((ARTIFACTS/'classification_model_stats.json').read_text())
        cls.samples=json.loads((ARTIFACTS/'classification_sample_orders.json').read_text())

    def setUp(self):
        from deployment.classification_streamlit import model_info
        model_info.clear()
        self.get_patch=patch('requests.get',side_effect=self.get_response)
        self.get=self.get_patch.start();self.addCleanup(self.get_patch.stop)
        self.app=AppTest.from_file(str(Path(__file__).with_name('classification_streamlit.py'))).run(timeout=20)

    def get_response(self,url,**kwargs):
        if url.endswith('/health'):return Response({'status':'healthy','model_loaded':True})
        if url.endswith('/model-info'):return Response(self.stats)
        if url.endswith('/samples'):return Response({'model_version':self.stats['model_version'],'samples':self.samples})
        if '/replay' in url:return Response({'available_days':['2018-06-01'],'orders':self.samples[:2]})
        if '/outcomes' in url:return Response({'outcomes':{self.samples[0]['order_id']:1,self.samples[1]['order_id']:None}})
        if '/logs/summary' in url:return Response({'request_count':0})
        raise AssertionError(url)

    def widget(self,collection,label):
        return next(x for x in getattr(self.app,collection) if x.label==label)

    def result(self,probability=.064):
        return {'late_probability':probability,'model_code':'C3','model_version':self.stats['model_version'],'drift_warnings':[],'is_predicted_late':False,'threshold_policy':'Illustrative 0.5 cutoff'}

    def test_trained_dropdowns_and_integer_counts(self):
        self.assertFalse(self.app.exception)
        for name in self.stats['features']['categorical']:
            from deployment.classification_streamlit import LABELS
            widget=self.widget('selectbox',LABELS[name][0])
            self.assertEqual(widget.options,self.stats['categories'][name])
        from deployment.classification_streamlit import COUNTS,LABELS
        for name in COUNTS:
            widget=self.widget('number_input',LABELS[name][0])
            self.assertIsInstance(widget.value,int)
            self.assertEqual(widget.step,1)

    def test_prediction_payload_exact_keys_and_types(self):
        with patch('requests.post',return_value=Response(self.result())) as post:
            self.widget('button','Check delivery risk').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(post.call_args.args[0].endswith('/predict'))
        features=post.call_args.kwargs['json']['features']
        self.assertEqual(set(features),set(self.stats['features']['numeric']+self.stats['features']['categorical']))
        from deployment.classification_streamlit import INTEGERS,BINARY
        for name in INTEGERS|BINARY:self.assertIsInstance(features[name],int)
        self.assertEqual(self.app.metric[0].value,'6.4%')

    def test_sample_switch_resets_widgets_even_when_switching_back(self):
        field=self.widget('number_input','Payment value (R$)')
        original=field.value
        field.set_value(original+50).run(timeout=20)
        picker=self.widget('selectbox','Held-out historical order')
        first,second=picker.options[:2]
        picker.select(second).run(timeout=20)
        self.widget('selectbox','Held-out historical order').select(first).run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertEqual(self.widget('number_input','Payment value (R$)').value,original)

    def test_invalid_calendar_blocks_submission(self):
        self.widget('number_input','Approval month').set_value(2)
        self.widget('number_input','Approval day of month').set_value(31).run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(self.widget('button','Check delivery risk').disabled)
        self.assertTrue(any('Invalid approval day/month' in e.value for e in self.app.error))

    def test_extrapolation_warning_and_numeric_null(self):
        self.widget('number_input','Payment value (R$)').set_value(1e10).run(timeout=20)
        self.assertTrue(any('outside refit range' in e.value for e in self.app.warning))
        field_name='payment_value_sum'
        sample_id=self.app.selectbox[0].value
        next(c for c in self.app.checkbox if c.key==f'input_{sample_id}_{field_name}_missing').check().run(timeout=20)
        with patch('requests.post',return_value=Response(self.result())) as post:
            self.widget('button','Check delivery risk').click().run(timeout=20)
        self.assertIsNone(post.call_args.kwargs['json']['features'][field_name])

    def test_http_errors_visible(self):
        with patch('requests.post',return_value=Response({'detail':'Invalid payment value'},422)):
            self.widget('button','Check delivery risk').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('422: Invalid payment value' in e.value for e in self.app.error))
        self.get.side_effect=requests.ConnectionError('backend unavailable')
        self.app.run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('backend unavailable' in e.value for e in self.app.error))

    def test_advanced_invalid_json_blocks_request(self):
        self.widget('checkbox','Use raw JSON instead of the form').check()
        self.app.text_area[0].set_value('{').run(timeout=20)
        with patch('requests.post') as post:
            self.app.run(timeout=20)
        self.assertTrue(self.widget('button','Check delivery risk').disabled)
        post.assert_not_called()
        self.assertFalse(self.app.exception)

    def test_all_pages_render_and_replay_outcomes_unknown(self):
        for page in ['How to read the score','Model performance','About','Daily watch-list replay']:
            self.app.radio[0].set_value(page).run(timeout=20)
            self.assertFalse(self.app.exception,page)
        response={'results':[{'index':i,'prediction':self.result(.1-i*.03)} for i in range(2)]}
        with patch('requests.post',return_value=Response(response)) as post:
            self.widget('button','Score this day').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertIn('rows',post.call_args.kwargs['json'])
        self.widget('checkbox','Show what actually happened (historical backtest)').check().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(any('1 unknown outcomes' in c.value for c in self.app.caption))

    def test_sensitivity_and_what_if_use_http(self):
        with patch('requests.post',return_value=Response(self.result())):
            self.widget('button','Check delivery risk').click().run(timeout=20)
        with patch('requests.post',return_value=Response({'sensitivities':[{'feature':'payment_value_sum','probability_change':.01}]})) as post:
            self.widget('button','Why this score?').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(post.call_args.args[0].endswith('/explain'))
        def batch(url,**kwargs):
            return Response({'results':[{'prediction':self.result()} for _ in kwargs['json']['rows']]})
        with patch('requests.post',side_effect=batch) as post:
            self.widget('button','Compare scenarios').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(post.call_args.args[0].endswith('/predict/batch'))

    def test_huge_numeric_json_shows_error_without_crash(self):
        self.widget('checkbox','Use raw JSON instead of the form').check()
        features={k:self.samples[0][k] for k in self.stats['features']['numeric']+self.stats['features']['categorical']}
        features['payment_value_sum']=10**500
        self.app.text_area[0].set_value(json.dumps(features)).run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertTrue(self.widget('button','Check delivery risk').disabled)

    def test_all_missing_numeric_features_blocked(self):
        from deployment.classification_streamlit import validate_form
        sample={k:self.samples[0][k] for k in self.stats['features']['numeric']+self.stats['features']['categorical']}
        for name in self.stats['features']['numeric']: sample[name]=None
        self.assertTrue(any('at least one numeric' in error for error in validate_form(sample,self.stats)))

    def test_replay_can_open_ranked_order(self):
        self.app.radio[0].set_value('Daily watch-list replay').run(timeout=20)
        response={'results':[{'index':i,'prediction':self.result(.1-i*.03)} for i in range(2)]}
        with patch('requests.post',return_value=Response(response)):
            self.widget('button','Score this day').click().run(timeout=20)
        selected=self.widget('selectbox','Inspect a ranked order').value
        self.widget('button','Open selected order').click().run(timeout=20)
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio[0].value,'Check an order')
        self.assertEqual(self.widget('selectbox','Held-out historical order').value,selected)

    def test_frontend_import_isolation(self):
        source=Path(__file__).with_name('classification_streamlit.py').read_text()
        self.assertNotIn('classification_service',source)
        self.assertNotIn('import sklearn',source)
        self.assertNotIn('import joblib',source)
        if os.environ.get('FRONTEND_ONLY_TEST')=='1':
            self.assertIsNone(importlib.util.find_spec('sklearn'))
            self.assertIsNone(importlib.util.find_spec('joblib'))
            self.assertNotIn('deployment.classification_service',sys.modules)


if __name__=='__main__':unittest.main()
