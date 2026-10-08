import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
from src import evaluation as ev, model_figures as mf


def toy_results():
    rows, folds = [], []
    for model, base in [('C1', .20), ('C3', .25), ('C3d', .22)]:
        for split in ev.SPLIT_ORDER:
            if split != 'CV':
                rows.append({'model': model, 'split': split, 'avg_precision': base + (.1 if split == 'Train' else 0),
                             'top10_lift': 2.5, 'precision_top10': base})
        folds += [{'model': model, 'fold': k, 'avg_precision': base + .01 * k, 'top10_lift': 2.0,
                   'precision_top10': base} for k in (1, 2, 3)]
    return pd.DataFrame(rows), pd.DataFrame(folds)


class TableTests(unittest.TestCase):
    def test_comparison_gap_and_tuning_tables(self):
        results, folds = toy_results()
        table = ev.comparison_table(results, folds, 'avg_precision', percent=True)
        self.assertIn('CV mean +/- SD', table.columns)
        gaps = ev.overfitting_gaps(results, folds, ['avg_precision']).set_index('model')
        self.assertAlmostEqual(gaps.loc['C1', 'avg_precision'], .30 - .22)
        gain = ev.tuning_gain_table(results, folds, 'C3', 'C3d', ['avg_precision'])
        self.assertAlmostEqual(gain.loc[0, 'June: gain'], .03)

    def test_success_criteria(self):
        results, _ = toy_results()
        table = ev.classification_success_criteria(results, ['C1'], ['C3'])
        rf = table[table['criterion'].str.contains('precision')]
        self.assertTrue(rf['passed'].all())

    def test_drift_table(self):
        results, _ = toy_results()
        drift = ev.drift_table(results, ['avg_precision'])
        self.assertEqual(len(drift), 3)

    def test_greedy_transform_cv_keeps_only_improving_group(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(0)
        X = pd.DataFrame({'x': rng.normal(size=300)})
        y = pd.Series(X['x'] ** 2 + rng.normal(scale=.1, size=300))
        folds = [(np.arange(0, 150 + 30 * k), np.arange(150 + 30 * k, 180 + 30 * k)) for k in range(5)]

        def build(kept):
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import FunctionTransformer
            func = (lambda d: d.assign(sq=d['x'] ** 2)) if 'squared' in kept else (lambda d: d)
            return make_pipeline(FunctionTransformer(func), LinearRegression())

        table, kept = ev.greedy_transform_cv(build, ['squared'], X, y, folds, 'r2')
        self.assertEqual(kept, ['squared'])


class FigureTests(unittest.TestCase):
    def test_figures_are_saved(self):
        path = pd.DataFrame({'step': [0, 1, 2], 'added_unit': ['(intercept only)', 'a', 'b'],
                             'cv_mean': [.1, .2, .21], 'cv_se': [.01, .01, .01]})
        deciles = {'C1': pd.DataFrame({'decile': range(1, 11), 'cumulative_capture_pct': np.linspace(30, 100, 10),
                                       'lift': np.linspace(3, 0.5, 10)})}
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            mf.plot_stepwise_path(path, 1, 2, 'CV AP', out / 'a.png')
            mf.plot_decile_lift(deciles, out / 'b.png')
            mf.plot_calibration({'Validation': {'C1': (y, rng.random(200))}}, out / 'c.png')
            mf.plot_model_comparison(pd.DataFrame({'Train': [.3], 'June': [.2]}, index=['C1']), 'AP', save_path=out / 'd.png')
            mf.plot_pr_curves({'Validation': {'C1': (y, rng.random(200))}}, out / 'e.png')
            self.assertEqual(len(list(out.glob('*.png'))), 5)


if __name__ == '__main__':
    unittest.main()
