import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')
from src import evaluation as ev, model_figures as mf
from src.regression_models import (ClippedDaysScorer, PromiseBaseline, clip_to_waiting_period,
                                   default_rf_pipeline, heteroscedasticity_table)


def toy_regression_results():
    rows = []
    for model, rmse in [('R0a', 10.0), ('R0b', 9.0), ('R1', 8.0), ('R2', 8.2), ('R3', 7.0)]:
        for split in ['June', 'Validation']:
            rows.append({'model': model, 'split': split, 'rmse': rmse, 'rmse_late': rmse * 2,
                         'mae_late': 1.0, 'rmse_on_time': 1.0, 'mae_on_time': 0.5,
                         'late_flag_precision': 0.5, 'late_flag_recall': 0.2, 'late_flag_f1': 0.3})
    return pd.DataFrame(rows)


class ModelTests(unittest.TestCase):
    def test_clipping_keeps_lead_time_within_zero_and_waiting_period(self):
        clipped = clip_to_waiting_period(np.array([-50.0, 0.0, 80.0]), np.array([10, 10, 10]))
        self.assertEqual(clipped.tolist(), [-10.0, 0.0, 35.0])

    def test_promise_baseline_predicts_zero_days(self):
        X = pd.DataFrame({'promised_lead_days': [10, 60]})
        self.assertEqual(PromiseBaseline().fit(X, [1, 2]).predict(X).tolist(), [0.0, -15.0])

    def test_scorer_adapter_returns_clipped_days_in_positive_column(self):
        X = pd.DataFrame({'promised_lead_days': [10.0, 10.0], 'x': [1.0, 100.0]})
        fitted = LinearRegression().fit(X[['x']], [1.0, 100.0])

        class Wrapper:
            def predict(self, frame):
                return fitted.predict(frame[['x']]) * 2

        scorer = ClippedDaysScorer(Wrapper())
        proba = scorer.predict_proba(X)
        self.assertEqual(list(scorer.classes_).index(1), 1)
        np.testing.assert_allclose(proba[:, 1], [2.0, 35.0])

    def test_default_rf_uses_library_defaults(self):
        forest = default_rf_pipeline(['a'], []).named_steps['model']
        self.assertEqual((forest.n_estimators, forest.max_depth, forest.min_samples_leaf), (100, None, 1))

    def test_heteroscedasticity_detects_growing_spread(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1, 10, 2000)
        y = 2 * x + rng.normal(scale=x)
        result = sm.OLS(y, sm.add_constant(x)).fit()
        summary, groups = heteroscedasticity_table(result)
        self.assertLess(summary.loc[0, 'p-value'], 0.001)
        self.assertGreater(groups['residual_sd'].iloc[-1], groups['residual_sd'].iloc[0])


class EvaluationTests(unittest.TestCase):
    def test_cv_regression_metrics_clips_predictions(self):
        X = pd.DataFrame({'promised_lead_days': [5.0] * 20, 'x': np.arange(20.0)})
        y = pd.Series(np.arange(20.0))
        folds = [(np.arange(10), np.arange(10, 20))]
        scores = ev.cv_regression_metrics(LinearRegression(), X, y, folds)
        self.assertEqual(list(scores.columns), ['fold'] + ev.REGRESSION_CV_METRICS)
        self.assertAlmostEqual(scores.loc[0, 'rmse'], 0.0, places=6)

    def test_score_regression_split_drops_unknown_targets(self):
        row = ev.score_regression_split('R1', 'June', pd.array([1.0, -2.0, None, 3.0], dtype='Float64'),
                                        [1.0, 1.0, 5.0, -1.0], pd.array([1, 0, None, 1], dtype='Int64'))
        self.assertEqual((row['n_scored'], row['n_unknown']), (3, 1))
        self.assertAlmostEqual(row['late_rate'], 2 / 3)
        self.assertAlmostEqual(row['late_flag_precision'], 0.5)
        self.assertAlmostEqual(row['late_flag_recall'], 0.5)

    def test_success_criteria_reports_relative_gains(self):
        table = ev.regression_success_criteria(toy_regression_results(), ['R1', 'R2'], ['R3'])
        r1 = table[(table['model'] == 'R1') & (table['split'] == 'June')].set_index('criterion')
        self.assertAlmostEqual(r1.loc['RMSE >= 10% below R0a', 'value'], 0.2)
        self.assertTrue(r1['passed'].all())
        report = table[table['criterion'].str.startswith('Relative RMSE gain')].iloc[0]
        self.assertAlmostEqual(report['value'], 1 - 7.0 / 8.0)
        self.assertIsNone(report['passed'])

    def test_error_and_late_flag_tables_have_split_columns(self):
        results = toy_regression_results()
        self.assertEqual(ev.error_split_table(results).columns.nlevels, 2)
        self.assertIn(('June', 'F1'), ev.late_flag_table(results).columns)


class FigureTests(unittest.TestCase):
    def test_regression_figures_are_saved(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(size=n)
        y = 3 * x + rng.normal(size=n)
        design = pd.DataFrame({'x': x})
        result = sm.OLS(y, sm.add_constant(design)).fit()
        bins = mf.partial_residual_bins(result, design, y, 'x')
        self.assertAlmostEqual(np.polyfit(bins['x'], bins['partial'], 1)[0], 3, delta=0.5)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / 'a.png', Path(folder) / 'b.png']
            mf.plot_predicted_vs_actual({'R1': (y, 3 * x)}, paths[0])
            mf.plot_ols_diagnostics(result.fittedvalues, result.resid, rng.choice(['1. A', '2. B'], n),
                                    rng.choice(['SP', 'RJ', 'MG'], n), paths[1])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == '__main__':
    unittest.main()
