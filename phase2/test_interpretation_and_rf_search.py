import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.interpretation import (agreement_table, assert_coefficients_match, coefficient_table,
                                fit_statsmodels, hypothesis_check_table)
from src.linear_transforms import linear_design, make_linear_preprocessor
from src.regression_models import PromiseBaseline, clip_to_waiting_period
from src.rf_search import _pick_best, search_space, stage_b_grid, tuned_rf_pipeline
from src.tuning import plain_logistic


def toy(n=800, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({'promised_lead_days': rng.uniform(5, 40, n),
                          'weight': rng.lognormal(6, 1, n),
                          'customer_state': rng.choice(['SP', 'RJ'], n)})
    index = -0.1 * frame['promised_lead_days'] + 0.0005 * frame['weight'] + 1.0 * (frame['customer_state'] == 'RJ')
    return frame, index


class InterpretationTests(unittest.TestCase):
    def fit(self, task):
        frame, index = toy()
        rng = np.random.default_rng(1)
        y = index + rng.normal(size=len(frame)) if task == 'regression' else (index + rng.logistic(size=len(frame)) > 0).astype(int)
        num, cat = ['promised_lead_days', 'weight'], ['customer_state']
        design, mapping, pre = linear_design(frame, num, cat, log_cols=['weight'])
        result = fit_statsmodels(task, design, y)
        model = plain_logistic() if task == 'classification' else LinearRegression()
        pipeline = Pipeline([('preprocessor', pre), ('model', model)]).fit(frame, y)
        return pipeline, result, pre, cat

    def test_statsmodels_and_sklearn_coefficients_match(self):
        for task in ['regression', 'classification']:
            with self.subTest(task=task):
                pipeline, result, _, _ = self.fit(task)
                self.assertLess(assert_coefficients_match(pipeline, result, tol=1e-3), 1e-3)
                shifted = result.params.copy()
                shifted.iloc[1] += 0.5
                with self.assertRaises(AssertionError):
                    assert_coefficients_match(pipeline, SimpleNamespace(params=shifted), tol=1e-3)

    def test_coefficient_table_scales_and_odds_ratios(self):
        pipeline, result, pre, cat = self.fit('classification')
        table = coefficient_table(result, 'classification', pre, cat).set_index('term')
        self.assertEqual(table.loc['log_weight', 'per_unit_basis'], 'per doubling')
        self.assertEqual(table.loc['customer_state_RJ', 'per_unit_basis'], 'vs reference level')
        self.assertAlmostEqual(table.loc['promised_lead_days', 'effect_per_sd'],
                               np.exp(table.loc['promised_lead_days', 'coef']))
        sd = table.loc['promised_lead_days', 'sd']
        self.assertAlmostEqual(table.loc['promised_lead_days', 'effect_per_unit'],
                               np.exp(table.loc['promised_lead_days', 'coef'] / sd))
        self.assertTrue((table[['ci_low', 'ci_high']].assign(c=table['coef'])
                         .apply(lambda r: r.ci_low < r.c < r.ci_high, axis=1)).all())

    def test_hypothesis_table_reports_sign_agreement_and_missing_features(self):
        _, result, pre, cat = self.fit('regression')
        table = coefficient_table(result, 'regression', pre, cat)
        hypotheses = {'promise': {'promised_lead_days': -1, 'weight': -1}, 'order': {'order_item_count': 1}}
        check = hypothesis_check_table({'R2': table}, hypotheses).set_index('feature')
        self.assertEqual(check.loc['promised_lead_days', 'verdict'], 'agree')
        self.assertEqual(check.loc['weight', 'verdict'], 'disagree')  # Truly positive, expected negative
        self.assertEqual(check.loc['order_item_count', 'verdict'], 'not in model')

    def test_agreement_table(self):
        importance = pd.DataFrame({'rank': [1, 2], 'feature': ['a', 'b'], 'importance_mean': [.2, .1]})
        coefs = pd.DataFrame({'unit': ['a', 'a', '(intercept)'], 'p_value': [.5, .01, 0.]})
        table = agreement_table(importance, coefs)
        self.assertTrue(table.loc[0, 'linear_significant'])
        self.assertTrue(pd.isna(table.loc[1, 'linear_significant']))


class RegressionModelTests(unittest.TestCase):
    def test_clip_and_promise_baseline_respect_the_waiting_period(self):
        promised = np.array([10., 40.])
        np.testing.assert_allclose(clip_to_waiting_period(np.array([-50., 50.]), promised), [-10., 5.])
        X = pd.DataFrame({'promised_lead_days': promised})
        np.testing.assert_allclose(PromiseBaseline().fit(X, [0, 0]).predict(X), [0., 0.])


class RFSearchTests(unittest.TestCase):
    def test_spaces_follow_the_plan(self):
        classifier, regressor = search_space('classification'), search_space('regression')
        self.assertEqual(set(classifier) - set(regressor), {'class_weight', 'criterion'})
        self.assertEqual(classifier['min_samples_leaf'], [1, 2, 5, 10, 20, 50, 100, 200])
        self.assertIn(None, classifier['max_depth'])
        self.assertEqual(len(classifier['max_depth']), 8)

    def test_stage_b_grid_uses_neighbours_and_tie_goes_to_simpler_forest(self):
        best = {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 5,
                'max_samples': 0.7}
        grid = stage_b_grid('regression', best)
        self.assertEqual(grid['max_depth'], [30, None])
        self.assertEqual(grid['min_samples_leaf'], [1, 2])
        self.assertEqual(grid['max_features'], ['log2', 'sqrt', 0.2])
        self.assertEqual(grid['n_estimators'], [300, 500])
        self.assertEqual(grid['min_samples_split'], [5])
        results = pd.DataFrame({'max_depth': [None, 8.0], 'min_samples_leaf': [1, 20],
                                'n_estimators': [500, 300], 'mean_score': [0.30000, 0.29995]})
        self.assertEqual(_pick_best(results, 1e-3)['max_depth'], 8.0)
        self.assertTrue(pd.isna(_pick_best(results, 1e-6)['max_depth']))

    def test_tuned_pipeline_applies_parameters(self):
        params = {'max_depth': 6.0, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'n_estimators': 300,
                  'class_weight': 'balanced', 'criterion': 'gini', 'min_samples_split': 2, 'max_samples': None}
        pipeline = tuned_rf_pipeline('classification', params, ['a'], [])
        forest = pipeline.named_steps['classifier']
        self.assertEqual((forest.max_depth, forest.n_estimators, forest.class_weight), (6, 300, 'balanced'))


if __name__ == '__main__':
    unittest.main()
