import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.splits import (day_blocked_time_series_folds, available_outcome_folds,
                        chronological_split, available_training_rows)
from src.labels import labels_as_of, regression_targets_as_of
from src.linear_transforms import make_linear_preprocessor
from src.tuning import plain_logistic, fit_logistic_checked


class PlainLogisticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame({'x': rng.normal(size=120)})
        self.y = pd.Series((self.X['x'] + rng.normal(size=120) > 0.8).astype(int))
        self.folds = [(np.arange(60), np.arange(60, 90)), (np.arange(90), np.arange(90, 120))]

    def test_plain_logistic_is_unweighted_and_unpenalised(self):
        logistic = plain_logistic()
        self.assertIsNone(logistic.class_weight)
        self.assertEqual((logistic.solver, logistic.max_iter), ('lbfgs', 5000))
        self.assertTrue(np.isinf(logistic.C) or logistic.penalty is None)

    def test_plain_logistic_matches_unpenalised_fit_and_fallback_is_l2(self):
        fitted = plain_logistic().fit(self.X, self.y)
        ridge = plain_logistic(ridge_fallback=True)
        self.assertEqual(ridge.C, 1e6)
        np.testing.assert_allclose(ridge.fit(self.X, self.y).coef_, fitted.coef_, rtol=1e-2)

    def test_logistic_checked_reports_converged_fit(self):
        pipeline = Pipeline([('preprocessor', make_linear_preprocessor(['x'], [])),
                             ('classifier', plain_logistic())])
        fitted, info = fit_logistic_checked(pipeline, self.X, self.y)
        self.assertTrue(info['converged'])
        self.assertEqual(len(fitted.predict_proba(self.X)), 120)


class TimeSeriesFoldTests(unittest.TestCase):
    def test_holdout_excludes_45_days_and_keeps_known_targets(self):
        rows = pd.DataFrame({'order_approved_dt': pd.date_range('2018-01-01', periods=180)})
        train, holdout = chronological_split(rows, test_days=45, gap_days=45)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(holdout), 45)
        self.assertGreater((holdout['order_approved_dt'].min() - train['order_approved_dt'].max()).days, 45)
        folds = day_blocked_time_series_folds(train['order_approved_dt'], n_splits=5,
                                              gap_days=45, min_train_days=30)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(len(tr) and len(va) for tr, va in folds))
        train = train.iloc[:3].assign(y=[0, 1, 1])
        def audit(r, date):
            self.assertEqual(date, holdout['order_approved_dt'].min().strftime('%Y-%m-%d'))
            return r.assign(known=pd.Series([0, pd.NA, 0], index=r.index, dtype='Int64'))
        available = available_training_rows(train, holdout['order_approved_dt'].min(),
                                             'y', audit, 'known')
        self.assertEqual(available.index.tolist(), [0])

    def test_folds_validate_on_later_days_and_never_split_a_day(self):
        dates = pd.Series(pd.date_range('2018-01-01', periods=12).repeat(3))
        for train_idx, valid_idx in day_blocked_time_series_folds(dates, n_splits=3):
            self.assertLess(dates.iloc[train_idx].max(), dates.iloc[valid_idx].min())
            self.assertFalse(set(dates.iloc[train_idx]) & set(dates.iloc[valid_idx]))

    def test_gap_and_initial_period_keep_all_five_regression_folds(self):
        dates = pd.Series(pd.date_range('2018-01-01', periods=135).repeat(2))
        folds = day_blocked_time_series_folds(dates, n_splits=5, gap_days=45,
                                              min_train_days=45)
        self.assertEqual(len(folds), 5)
        for tr, va in folds:
            self.assertGreater((dates.iloc[va].min() - dates.iloc[tr].max()).days, 45)
            self.assertGreaterEqual(len(tr), 90)
            self.assertFalse(set(dates.iloc[tr]) & set(dates.iloc[va]))

    def test_fixed_calendar_validation_windows_are_30_days_and_anchored_at_end(self):
        dates = pd.Series(pd.date_range('2017-04-18', periods=290).repeat(2))
        dates = dates.loc[dates.ne(pd.Timestamp('2017-09-20'))].reset_index(drop=True)
        folds = day_blocked_time_series_folds(
            dates, n_splits=5, gap_days=45, validation_days=30)

        self.assertEqual(len(folds), 5)
        expected_first_valid = dates.max() - pd.Timedelta(days=5 * 30 - 1)
        self.assertEqual(dates.iloc[folds[0][1]].min(), expected_first_valid)
        self.assertEqual((dates.iloc[folds[0][1]].max() - expected_first_valid).days, 29)
        self.assertEqual(dates.iloc[folds[0][1]].nunique(), 29)
        self.assertEqual((dates.iloc[folds[0][0]].max() - dates.iloc[folds[0][0]].min()).days + 1, 95)

        for train_idx, valid_idx in folds:
            valid_dates = dates.iloc[valid_idx]
            self.assertEqual((valid_dates.max() - valid_dates.min()).days + 1, 30)
            self.assertGreater((valid_dates.min() - dates.iloc[train_idx].max()).days, 45)
            self.assertFalse(set(dates.iloc[train_idx]) & set(valid_dates))

        self.assertTrue(all(len(folds[i][0]) < len(folds[i + 1][0]) for i in range(4)))

    def test_fixed_windows_reject_conflicting_initial_period(self):
        dates = pd.date_range('2018-01-01', periods=300)
        with self.assertRaisesRegex(ValueError, 'Use validation_days or min_train_days'):
            day_blocked_time_series_folds(
                dates, n_splits=5, gap_days=45, min_train_days=95, validation_days=30)

    def test_fixed_windows_reject_no_dates_or_no_initial_training_history(self):
        with self.assertRaisesRegex(ValueError, 'at least one row'):
            day_blocked_time_series_folds([], n_splits=5, gap_days=45, validation_days=30)

        dates = pd.date_range('2018-01-01', periods=45)
        with self.assertRaisesRegex(ValueError, 'Insufficient history'):
            day_blocked_time_series_folds(
                dates, n_splits=5, gap_days=45, validation_days=30)

    def test_empty_folds_raise_instead_of_disappearing(self):
        dates = pd.date_range('2018-01-01', periods=135)
        with self.assertRaisesRegex(ValueError, 'Empty CV fold'):
            day_blocked_time_series_folds(dates, n_splits=5, gap_days=45)

    def test_classification_excludes_unknown_labels_despite_approval_gap(self):
        rows = pd.DataFrame({
            'order_approved_at': pd.to_datetime(['2018-01-01'] * 3 + ['2018-03-01']),
            'order_delivered_customer_date': pd.to_datetime(['2018-01-10', '2018-03-10', None, None]),
            'order_estimated_delivery_date': pd.to_datetime(['2018-01-20', '2018-03-20', '2018-02-01', '2018-03-20'])})
        rows['order_approved_dt'] = rows['order_approved_at']
        targets = labels_as_of(rows, '2018-04-01')['label_as_of_run']
        folds = available_outcome_folds(rows, [(np.arange(3), np.array([3]))],
                                        targets, labels_as_of, 'label_as_of_run')
        np.testing.assert_array_equal(folds[0][0], [0, 2])
        audit = labels_as_of(rows.iloc[folds[0][0]], '2018-03-01')
        self.assertTrue(audit['label_as_of_run'].notna().all())

    def test_regression_capped_outcome_is_known_without_delivery(self):
        rows = pd.DataFrame({
            'order_approved_at': pd.to_datetime(['2018-01-01', '2018-02-20', '2018-03-01']),
            'order_delivered_customer_date': pd.to_datetime(['2018-03-10', '2018-03-05', None])})
        rows['order_approved_dt'] = rows['order_approved_at']
        rows['order_estimated_delivery_date'] = rows['order_approved_at'] + pd.Timedelta(days=20)
        targets = regression_targets_as_of(rows, '2018-04-01', 45)['target_as_of_run']
        folds = available_outcome_folds(
            rows, [(np.arange(2), np.array([2]))], targets,
            lambda r, d: regression_targets_as_of(r, d, 45), 'target_as_of_run')
        np.testing.assert_array_equal(folds[0][0], [0])
        self.assertEqual(regression_targets_as_of(rows.iloc[[0]], '2018-03-01', 45)
                         ['target_as_of_run'].iloc[0], 45 - 20)  # Capped lead time minus the promise


if __name__ == '__main__':
    unittest.main()
