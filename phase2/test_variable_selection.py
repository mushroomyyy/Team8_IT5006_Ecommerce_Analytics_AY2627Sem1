import unittest
from functools import partial

import numpy as np
import pandas as pd

from src.linear_transforms import linear_design
from src.splits import day_blocked_time_series_folds
from src.variable_selection import (check_train_only, forward_stepwise_cv, forward_stepwise_ic,
                                    lasso_units, make_unit_pipeline_builder, one_se_rule,
                                    selection_overlap, units_through_step)

NUM = ['signal', 'noise_a', 'noise_b', 'noise_c']
CAT = ['customer_state']
UNITS = NUM + CAT


def toy_rows(task, informative_state=False, n_days=300, per_day=6, seed=1):
    """Rows with one pure-signal feature, three noise features and one categorical."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    rows = pd.DataFrame({
        'order_id': [f'o{i}' for i in range(n)],
        'order_approved_dt': pd.Timestamp('2018-01-01') + pd.to_timedelta(np.repeat(np.arange(n_days), per_day), unit='D'),
        **{c: rng.normal(size=n) for c in NUM},
        'customer_state': rng.choice(['SP', 'RJ', 'MG'], n),
    })
    state_effect = rows['customer_state'].map({'SP': 0.0, 'RJ': 1.5, 'MG': -1.5}) if informative_state else 0
    index = 2.0 * rows['signal'] + state_effect
    if task == 'regression':
        y = index + rng.normal(size=n)
    else:
        y = (index + rng.logistic(size=n) > 1.0).astype(int)
    folds = day_blocked_time_series_folds(rows['order_approved_dt'], n_splits=3, gap_days=0,
                                          validation_days=30)
    return rows, pd.Series(y), folds


class StepwiseCVTests(unittest.TestCase):
    def run_cv(self, task, units=UNITS, **kwargs):
        rows, y, folds = toy_rows(task, **kwargs.pop('data', {}))
        builder = make_unit_pipeline_builder(task, NUM, CAT)
        scoring = 'average_precision' if task == 'classification' else 'neg_root_mean_squared_error'
        return forward_stepwise_cv(rows, y, units, builder, folds, scoring, set(rows['order_id']),
                                   n_jobs=1, **kwargs), (rows, y, folds)

    def test_pure_signal_feature_is_picked_first(self):
        for task in ['regression', 'classification']:
            with self.subTest(task=task):
                (path, candidates), _ = self.run_cv(task)
                self.assertEqual(path.loc[1, 'added_unit'], 'signal')
                self.assertEqual(path['step'].tolist(), list(range(len(UNITS) + 1)))
                self.assertEqual(path.loc[0, 'added_unit'], '(intercept only)')
                first = candidates[candidates['step'] == 1]
                self.assertEqual(len(first), len(UNITS))
                self.assertEqual(first.loc[first['chosen'], 'unit'].tolist(), ['signal'])

    def test_categorical_block_is_added_whole(self):
        (path, _), _ = self.run_cv('regression', data={'informative_state': True})
        step = path.index[path['added_unit'] == 'customer_state'][0]
        # Three states with SP as reference: two dummy columns enter together
        self.assertEqual(path.loc[step, 'n_encoded_columns'] - path.loc[step - 1, 'n_encoded_columns'], 2)
        self.assertLessEqual(step, 2)

    def test_deterministic(self):
        (first, cand_first), _ = self.run_cv('classification')
        (second, cand_second), _ = self.run_cv('classification')
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(cand_first, cand_second)

    def test_early_stop_truncates_and_logs(self):
        with self.assertLogs('src.variable_selection', level='WARNING'):
            (path, _), _ = self.run_cv('regression', early_stop=1)
        self.assertLess(len(path), len(UNITS) + 1)
        self.assertEqual(path.attrs['stopped_early_after_step'], path['step'].max())

    def test_rows_outside_train_are_rejected(self):
        rows, y, folds = toy_rows('regression')
        builder = make_unit_pipeline_builder('regression', NUM, CAT)
        train_ids = set(rows['order_id'].iloc[:-5])
        with self.assertRaisesRegex(ValueError, 'outside the declared Train'):
            forward_stepwise_cv(rows, y, UNITS, builder, folds, 'neg_root_mean_squared_error',
                                train_ids, n_jobs=1)
        with self.assertRaisesRegex(ValueError, 'beyond the Train rows'):
            check_train_only(rows.iloc[:100], set(rows['order_id']), folds)
        design_builder = lambda r: linear_design(r, NUM, CAT)[:2]
        with self.assertRaisesRegex(ValueError, 'outside the declared Train'):
            forward_stepwise_ic(rows, y, UNITS, design_builder, 'regression', train_ids)
        with self.assertRaisesRegex(ValueError, 'outside the declared Train'):
            lasso_units(rows, y, UNITS, design_builder, 'regression', folds, train_ids, n_jobs=1)


class OneSERuleTests(unittest.TestCase):
    def path(self, means, se=0.3):
        return pd.DataFrame({'step': range(len(means)), 'cv_mean': means, 'cv_se': se})

    def test_smallest_step_within_one_se_of_best(self):
        path = self.path([-10, -5, -3.2, -3.0, -3.05, -3.1])
        self.assertEqual(one_se_rule(path, higher_is_better=True), (2, 3))

    def test_lower_is_better_and_intercept_is_never_chosen(self):
        path = self.path([9.0, 5.0, 3.2, 3.0, 3.05])
        self.assertEqual(one_se_rule(path, higher_is_better=False), (2, 3))
        flat = self.path([3.0, 3.0, 3.0])
        self.assertEqual(one_se_rule(flat, higher_is_better=False), (1, 1))

    def test_units_through_step(self):
        path = pd.DataFrame({'step': [0, 1, 2, 3], 'added_unit': ['(i)', 'a', 'b', 'c']})
        self.assertEqual(units_through_step(path, 2), ['a', 'b'])


class StepwiseICTests(unittest.TestCase):
    def run_ic(self, task, criterion='bic', **data):
        rows, y, folds = toy_rows(task, **data)
        design_builder = lambda r: linear_design(r, NUM, CAT)[:2]
        result = forward_stepwise_ic(rows, y, UNITS, design_builder, task, set(rows['order_id']),
                                     criterion=criterion, n_jobs=1)
        return result, (rows, y, folds, design_builder)

    def test_bic_picks_signal_first_and_stops_on_noise(self):
        for task in ['regression', 'classification']:
            with self.subTest(task=task):
                (path, selected), _ = self.run_ic(task)
                self.assertEqual(selected, ['signal'])
                self.assertEqual(path['added_unit'].tolist(), ['(intercept only)', 'signal'])
                self.assertLess(path.loc[1, 'bic'], path.loc[0, 'bic'])

    def test_informative_categorical_enters_as_a_block_with_both_dummies(self):
        (path, selected), _ = self.run_ic('regression', informative_state=True)
        self.assertEqual(set(selected), {'signal', 'customer_state'})
        step = path.index[path['added_unit'] == 'customer_state'][0]
        self.assertEqual(path.loc[step, 'k'] - path.loc[step - 1, 'k'], 2)

    def test_aic_selects_at_least_as_many_units_as_bic(self):
        (_, bic_units), _ = self.run_ic('regression', 'bic')
        (_, aic_units), _ = self.run_ic('regression', 'aic')
        self.assertTrue(set(bic_units) <= set(aic_units))

    def test_nonconverging_logit_candidate_is_skipped_and_logged(self):
        rows, y, folds = toy_rows('classification', informative_state=True)
        y = y.copy()
        y[rows['customer_state'] == 'MG'] = 0  # Quasi-separation for the MG dummy
        design_builder = lambda r: linear_design(r, NUM, CAT)[:2]
        with self.assertLogs('src.variable_selection', level='WARNING'):
            path, selected = forward_stepwise_ic(
                rows, y, UNITS, design_builder, 'classification', set(rows['order_id']), n_jobs=1,
                fallback_method=None)
        skipped = path.attrs['skipped']
        self.assertTrue((skipped['unit'] == 'customer_state').any())
        self.assertNotIn('customer_state', selected)


class LassoAndOverlapTests(unittest.TestCase):
    def test_lasso_keeps_signal_and_whole_categorical_block(self):
        rows, y, folds = toy_rows('regression', informative_state=True)
        design_builder = lambda r: linear_design(r, NUM, CAT)[:2]
        result = lasso_units(rows, y, UNITS, design_builder, 'regression', folds,
                             set(rows['order_id']), n_jobs=1)
        self.assertIn('signal', result['selected_units'])
        self.assertIn('customer_state', result['selected_units'])

    def test_overlap_counts_methods(self):
        path = pd.DataFrame({'step': [0, 1, 2], 'added_unit': ['(i)', 'a', 'b']})
        table = selection_overlap(['a', 'b', 'c'], path, 1, bic_selected=['a'], aic_selected=['a', 'b'],
                                  lasso_selected=['a', 'c'])
        rows = table.set_index('unit')
        self.assertEqual(rows.loc['a', 'n_methods'], 4)
        self.assertEqual(rows.loc['b', 'n_methods'], 1)  # Entered at step 2 but beyond the chosen step 1
        self.assertEqual(rows.loc['b', 'cv_step_entered'], 2)
        self.assertTrue(pd.isna(rows.loc['c', 'cv_step_entered']))
        self.assertEqual(table['unit'].iloc[0], 'a')


if __name__ == '__main__':
    unittest.main()
