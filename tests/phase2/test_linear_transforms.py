import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import (classification_eval_metrics, fingerprint,
                            regression_eval_metrics)
from src.linear_transforms import (linear_design, make_linear_preprocessor, readable_names,
                                   skewness_table, unit_columns)
from src.tuning import plain_logistic


def toy_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'weight': rng.lognormal(mean=6, sigma=1.5, size=n),
        'order_approved_month': rng.integers(1, 13, n),
        'order_approved_day_of_week': rng.integers(0, 7, n),
        'is_weekend_approval': rng.integers(0, 2, n),
        'customer_state': rng.choice(['SP', 'RJ', 'MG'], n),
    })


NUM = ['weight', 'order_approved_month', 'order_approved_day_of_week', 'is_weekend_approval']
CAT = ['customer_state']


class LinearTransformTests(unittest.TestCase):
    def test_log1p_and_cyclic_columns_are_named_and_computed(self):
        frame = toy_frame()
        pre = make_linear_preprocessor(NUM, CAT, log_cols=['weight'], cyclic=True).fit(frame)
        names = readable_names(pre)
        self.assertEqual(names[:5], ['log_weight', 'order_approved_month_sin', 'order_approved_month_cos',
                                     'order_approved_day_of_week_sin', 'order_approved_day_of_week_cos'])
        self.assertNotIn('order_approved_month', names)
        self.assertEqual(names[-2:], ['customer_state_MG', 'customer_state_RJ'])  # SP is the reference
        design = pd.DataFrame(pre.transform(frame), columns=names)
        np.testing.assert_allclose(design['log_weight'].mean(), 0, atol=1e-9)
        expand = pre.named_steps['expand'].transform(frame)
        np.testing.assert_allclose(expand['log_weight'], np.log1p(frame['weight']))
        angle = 2 * np.pi * frame['order_approved_month'] / 12
        np.testing.assert_allclose(expand['order_approved_month_sin'], np.sin(angle))
        np.testing.assert_allclose(expand['order_approved_day_of_week_cos'],
                                   np.cos(2 * np.pi * frame['order_approved_day_of_week'] / 7))

    def test_everything_is_fitted_on_the_rows_passed_to_fit(self):
        train, later = toy_frame(), toy_frame(seed=5).assign(weight=1e7)
        pre = make_linear_preprocessor(NUM, CAT, log_cols=['weight']).fit(train)
        scale = pre.named_steps['encode'].named_transformers_['numeric'].named_steps['scaler'].mean_
        pre.transform(later)
        np.testing.assert_array_equal(
            pre.named_steps['encode'].named_transformers_['numeric'].named_steps['scaler'].mean_, scale)
        self.assertAlmostEqual(scale[0], np.log1p(train['weight']).mean())

    def test_options_off_reproduce_the_plain_design_and_subsets_are_supported(self):
        frame = toy_frame()
        names = readable_names(make_linear_preprocessor(NUM, CAT).fit(frame))
        self.assertEqual(names, NUM + ['customer_state_MG', 'customer_state_RJ'])
        subset = make_linear_preprocessor(['is_weekend_approval'], [], log_cols=['weight'],
                                          cyclic=True).fit(frame)
        self.assertEqual(readable_names(subset), ['is_weekend_approval'])

    def test_unit_columns_group_transformed_variants_with_their_raw_feature(self):
        frame = toy_frame()
        design, mapping, pre = linear_design(frame, NUM, CAT, log_cols=['weight'], cyclic=True)
        self.assertEqual(mapping['weight'], ['log_weight'])
        self.assertEqual(mapping['order_approved_month'],
                         ['order_approved_month_sin', 'order_approved_month_cos'])
        self.assertEqual(mapping['customer_state'], ['customer_state_MG', 'customer_state_RJ'])
        self.assertEqual(sum(len(v) for v in mapping.values()), design.shape[1])
        self.assertEqual(unit_columns(pre, CAT), mapping)

    def test_mixed_route_merges_into_interstate_only_when_requested(self):
        frame = toy_frame().assign(route_type=['1. All interstate', '2. All same-state', '3. Mixed'] * 20)
        merged = readable_names(make_linear_preprocessor([], ['route_type'], merge_mixed_route=True).fit(frame))
        self.assertEqual(merged, ['route_type_2. All same-state'])
        plain = readable_names(make_linear_preprocessor([], ['route_type']).fit(frame))
        self.assertEqual(plain, ['route_type_2. All same-state', 'route_type_3. Mixed'])

    def test_skewness_table_flags_skewed_non_negative_columns(self):
        frame = toy_frame().assign(signed=np.linspace(-5, 5, 60))
        table = skewness_table(frame, ['weight', 'signed']).set_index('feature')
        self.assertTrue(table.loc['weight', 'log1p_recommended'])
        self.assertLess(abs(table.loc['weight', 'skew_log1p']), abs(table.loc['weight', 'skew_raw']))
        self.assertFalse(table.loc['signed', 'log1p_recommended'])
        self.assertTrue(np.isnan(table.loc['signed', 'skew_log1p']))


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = pd.DataFrame({'x': rng.normal(size=300), 'z': rng.normal(size=300)})
        self.y = pd.Series((self.X['x'] + rng.normal(size=300) > 0.5).astype(int))

    def test_fingerprint_is_stable_for_identical_fits_and_scoring_but_changes_with_data(self):
        classifiers = {'Logistic Regression': plain_logistic(),
                       'Random Forest': RandomForestClassifier(n_estimators=20, random_state=0)}
        for name, classifier in classifiers.items():
            with self.subTest(model=name):
                model = Pipeline([('preprocessor', make_linear_preprocessor(['x', 'z'], [])),
                                  ('classifier', classifier)])
                first = clone(model).fit(self.X, self.y)
                second = clone(model).fit(self.X, self.y)
                before = fingerprint(first)
                self.assertEqual(before, fingerprint(second))
                first.predict_proba(self.X)
                self.assertEqual(before, fingerprint(first))
                refit = clone(model).fit(self.X.iloc[:200], self.y.iloc[:200])
                self.assertNotEqual(before, fingerprint(refit))


class EvaluationMetricTests(unittest.TestCase):
    def test_classification_metrics_use_a_top_decile_cutoff_of_scored_orders(self):
        prob = np.linspace(0.01, 0.99, 100)
        y = pd.array([0] * 90 + [1] * 10, dtype='Int64')
        metrics = classification_eval_metrics(y, prob)
        self.assertEqual(metrics['precision_top10'], 1.0)
        self.assertEqual(metrics['top10_capture'], 1.0)
        self.assertAlmostEqual(metrics['top10_lift'], 10.0)
        self.assertAlmostEqual(metrics['avg_precision'], 1.0)
        self.assertEqual(metrics['recall_at_0_5'], 1.0)
        self.assertAlmostEqual(metrics['brier'], np.mean((prob - np.asarray(y, dtype=float)) ** 2))

    def test_unknown_outcomes_stay_ranked_but_never_count_as_late(self):
        prob = np.linspace(0.01, 0.99, 20)
        y = pd.array([0] * 17 + [pd.NA, pd.NA, 1], dtype='Int64')
        metrics = classification_eval_metrics(y, prob)
        self.assertEqual(metrics['n_known'], 18)
        self.assertEqual(metrics['precision_top10'], 0.5)  # Top 2 scored: one late, one unknown

    def test_regression_metrics_split_by_actual_late_and_derive_a_flag(self):
        actual = np.array([-5., -2., 3., 6.])
        predicted = np.array([-4., 1., 2., -1.])
        metrics = regression_eval_metrics(actual, predicted)
        self.assertAlmostEqual(metrics['mae_late'], 4.0)  # |2-3| and |-1-6|
        self.assertAlmostEqual(metrics['mae_on_time'], 2.0)  # |-4+5| and |1+2|
        self.assertEqual(metrics['late_flag_precision'], 0.5)
        self.assertEqual(metrics['late_flag_recall'], 0.5)
        self.assertEqual(metrics['n_actual_late'], 2)


if __name__ == '__main__':
    unittest.main()
