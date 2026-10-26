import unittest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.evaluation import confusion_at, f1_threshold, false_negative_breakdown
from src.model_selection import (bootstrap_vs_leader, choose_model, model_size,
                                 paired_fold_differences)
from src.tree_search import build_tree_pipeline, run_tree_search, tuned_tree_pipeline


def toy_classification(n_days=30, per_day=60, seed=0):
    rng = np.random.default_rng(seed)
    days = np.repeat(pd.date_range('2018-03-19', periods=n_days), per_day)
    signal = rng.normal(size=len(days))
    y = (signal + rng.normal(scale=1.0, size=len(days)) > 1.2).astype(int)
    return y, signal, days, rng


class BootstrapSelectionTests(unittest.TestCase):
    def test_near_copy_is_tied_and_noise_is_not(self):
        y, signal, days, rng = toy_classification()
        preds = {'strong': signal, 'near_copy': signal + rng.normal(scale=0.01, size=len(y)),
                 'noise': rng.normal(size=len(y))}
        table = bootstrap_vs_leader(y, preds, days, 'classification', n_boot=300).set_index('model')
        self.assertTrue(table.loc['near_copy', 'tied'])
        self.assertFalse(table.loc['noise', 'tied'])
        self.assertEqual(table.attrs['n_units'], 30)
        self.assertTrue(table.loc[table.attrs['leader'], 'leader'])
        self.assertTrue((table['diff_vs_leader'] <= 0).all())

    def test_rmse_leader_is_lowest(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=600)
        preds = {'good': y + rng.normal(scale=0.5, size=600), 'bad': y + rng.normal(scale=2.0, size=600)}
        table = bootstrap_vs_leader(y, preds, None, 'regression', n_boot=200).set_index('model')
        self.assertEqual(table.attrs['leader'], 'good')
        self.assertGreater(table.loc['bad', 'diff_vs_leader'], 0)
        self.assertFalse(table.loc['bad', 'tied'])

    def test_simplest_tied_model_wins_and_baselines_are_excluded(self):
        table = pd.DataFrame({'model': ['C0', 'big', 'small', 'forest', 'loser'],
                              'tied': [True, True, True, True, False]})
        families = {'big': 'linear', 'small': 'linear', 'forest': 'forest', 'loser': 'linear'}
        sizes = {'big': 55, 'small': 29, 'forest': 9000, 'loser': 2}
        choice, out = choose_model(table, families, sizes)
        self.assertEqual(choice, 'small')
        decisions = out.set_index('model')['decision']
        self.assertTrue(decisions['C0'].startswith('baseline'))
        self.assertTrue(decisions['loser'].startswith('not tied'))
        self.assertTrue(decisions['big'].startswith('tied, but'))

    def test_family_beats_size(self):
        table = pd.DataFrame({'model': ['tree', 'linear'], 'tied': [True, True]})
        choice, _ = choose_model(table, {'tree': 'tree', 'linear': 'linear'}, {'tree': 4, 'linear': 55})
        self.assertEqual(choice, 'linear')

    def test_model_size_counts_coefficients_and_intercept(self):
        fitted = LinearRegression().fit(np.random.default_rng(0).normal(size=(50, 3)), np.arange(50.0))
        self.assertEqual(model_size(fitted, 'linear'), 4)


class PairedFoldTests(unittest.TestCase):
    def test_corrected_interval_is_wider_than_plain_t(self):
        diffs = np.array([0.02, 0.019, 0.004, 0.017, 0.022])
        folds_df = pd.DataFrame({'model': ['a'] * 5 + ['b'] * 5, 'fold': list(range(1, 6)) * 2,
                                 'avg_precision': np.r_[0.3 + diffs, np.full(5, 0.3)]})
        folds = [(np.arange(100 * (k + 1)), np.arange(30)) for k in range(5)]
        out = paired_fold_differences(folds_df, 'a', 'b', 'avg_precision', folds)
        self.assertAlmostEqual(out['mean_diff'], diffs.mean())
        plain = stats.t.ppf(0.975, 4) * diffs.std(ddof=1) / np.sqrt(5)
        self.assertGreater(out['ci_high'] - out['mean_diff'], plain)
        self.assertEqual(out['folds_better'], '5/5')

    def test_lower_is_better_counts_wins(self):
        folds_df = pd.DataFrame({'model': ['a'] * 3 + ['b'] * 3, 'fold': [1, 2, 3] * 2,
                                 'rmse': [7.0, 8.0, 9.5, 7.5, 8.5, 9.0]})
        folds = [(np.arange(100), np.arange(10))] * 3
        out = paired_fold_differences(folds_df, 'a', 'b', 'rmse', folds, higher_is_better=False)
        self.assertEqual(out['folds_better'], '2/3')


class ThresholdTests(unittest.TestCase):
    def test_f1_threshold_and_confusion(self):
        y = pd.Series([0, 0, 1, 1, 0, 1, pd.NA], dtype='Int64')
        prob = np.array([0.1, 0.2, 0.8, 0.7, 0.75, 0.3, 0.9])
        threshold = f1_threshold(y, prob)
        self.assertIn(threshold, prob[:6])
        counts = confusion_at(y, prob, threshold)
        self.assertEqual(sum(counts[k] for k in ['true_positive', 'false_positive', 'false_negative',
                                                 'true_negative']), 6)
        at_half = confusion_at(y, prob, 0.5)
        self.assertEqual((at_half['true_positive'], at_half['false_positive'], at_half['false_negative']), (2, 1, 1))

    def test_false_negative_breakdown(self):
        scored = pd.DataFrame({'order_id': list('abcde'), 'predicted_probability': [0.9, 0.1, 0.2, 0.05, 0.6],
                               'actual_label': [1, 1, 1, 0, 1]})
        features = pd.DataFrame({'order_id': list('abcde'), 'customer_state': ['SP', 'BA', 'BA', 'SP', 'SP'],
                                 'promised_lead_days': [10, 20, 30, 5, 25]})
        table = false_negative_breakdown(scored, features, 0.5, 'customer_state').set_index('customer_state')
        self.assertEqual(table.loc['BA', 'missed'], 2)
        self.assertEqual(table.loc['SP', 'late_orders'], 2)
        self.assertAlmostEqual(table.loc['BA', 'share_of_missed_pct'], 100.0)
        bands = false_negative_breakdown(scored, features, 0.5, 'promised_lead_band')
        self.assertEqual(int(bands['missed'].sum()), 2)


class TreeSearchTests(unittest.TestCase):
    def test_search_prefers_fewest_leaves_within_tolerance(self):
        rng = np.random.default_rng(0)
        n = 400
        frame = pd.DataFrame({'promised_lead_days': rng.uniform(5, 40, n), 'weight': rng.lognormal(6, 1, n),
                              'customer_state': rng.choice(['SP', 'RJ'], n),
                              'route_type': rng.choice(['1. All interstate', '2. All same-state'], n)})
        y = (frame['promised_lead_days'] < 12).astype(int)
        folds = [(np.arange(0, 200), np.arange(200, 300)), (np.arange(0, 300), np.arange(300, 400))]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tree_search('classification', frame, y, folds, ['promised_lead_days', 'weight'],
                                     ['customer_state', 'route_type'], output_dir=tmp, n_jobs=1)
        self.assertLessEqual(result['metadata']['best_n_leaves'], 4)
        pipeline = tuned_tree_pipeline('classification', result['best_params'],
                                       ['promised_lead_days', 'weight'], ['customer_state', 'route_type'])
        self.assertEqual(type(pipeline), type(build_tree_pipeline('classification', [], [])))


if __name__ == '__main__':
    unittest.main()
