"""Run: .venv/bin/python -m unittest discover -s phase2 -p 'test_*.py'."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model_workbench import (load_order_tables, build_order_features, build_order_dataset,
                             build_temporal_folds, coverage_metrics, FEATURES)
from classification_backtest import evaluate_out_of_fold_predictions

ARCHIVE = Path(__file__).resolve().parents[1] / 'streamlit_release/data/olist_csv.zip'


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_order_dataset(ARCHIVE)

    def test_training_labels_and_windows(self):
        for fold in build_temporal_folds(self.dataset):
            origin = pd.Timestamp(fold['validation_start'])
            train = fold['train']
            self.assertTrue(train.label_available_at.le(origin).all())
            self.assertTrue(train.approval_date.ge(origin-pd.Timedelta(days=210)).all())
            self.assertTrue(train.approval_date.lt(origin-pd.Timedelta(days=30)).all())
            self.assertFalse(set(train.order_id) & set(fold['validation'].order_id))
            pending = train.loc[train.order_delivered_customer_date.ge(origin)]
            self.assertTrue(pending.late_delivery.eq(1).all())
            self.assertGreater(len(pending), 0)

    def test_live_features_independent_of_outcomes(self):
        tables = load_order_tables(ARCHIVE)
        expected = build_order_features(tables, '2018-05-01')
        # Removing every actual outcome must preserve the entire feature frame.
        tables['orders'] = tables['orders'].drop(columns=[
            'order_status', 'order_delivered_customer_date', 'order_delivered_carrier_date'])
        actual = build_order_features(tables, '2018-05-01')
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(actual.order_id.nunique(), len(actual))
        self.assertEqual(set(actual.columns), {'order_id', 'approval_date', *FEATURES})

    def test_threshold_cannot_use_later_fold_or_unmatured_labels(self):
        p = pd.DataFrame(dict(actual=[0, 1, 0, 1, 0, 1],
            probability=[.2, .7, .3, .8, .4, .6], fold=[1, 1, 2, 2, 3, 3],
            approval_date=pd.to_datetime(['2018-01-01']*2+['2018-02-01']*2+['2018-03-01']*2),
            label_available_at=pd.to_datetime(['2018-01-20']*2+['2018-04-01']*2+['2018-04-02']*2)))
        result = evaluate_out_of_fold_predictions(p)
        audit = result['threshold_audit']
        self.assertEqual([a['history_rows'] for a in audit], [0, 2, 2])
        for a in audit:
            if a['latest_label']:
                self.assertLessEqual(pd.Timestamp(a['latest_label']), pd.Timestamp(a['origin']))
        changed = p.copy()
        changed.loc[changed.fold.ge(2), 'actual'] = 1-changed.loc[changed.fold.ge(2), 'actual']
        second = evaluate_out_of_fold_predictions(changed)
        self.assertEqual([a['threshold'] for a in audit],
                         [a['threshold'] for a in second['threshold_audit']])

    def test_daily_capacity_and_ties(self):
        # Constant probabilities must match same-day random selection exactly.
        y = pd.Series([0]*90+[1]*10)
        dates = np.repeat(['2018-01-01', '2018-01-02'], 50)
        a = coverage_metrics(y, np.full(100, .1), dates=dates)
        b = coverage_metrics(y.iloc[::-1], np.full(100, .1), dates=dates[::-1])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['lift_at_10pct'], 1.0)
        self.assertAlmostEqual(a['coverage_at_10pct'], .1)
        # Both positive orders rank first within their own day, despite score scales.
        daily = coverage_metrics(pd.Series([1,0,1,0]), np.array([.9,.8,.2,.1]),
                                 fractions=(.5,), dates=['a','a','b','b'])
        self.assertEqual(daily['coverage_at_50pct'], 1.0)


if __name__ == '__main__':
    unittest.main()
