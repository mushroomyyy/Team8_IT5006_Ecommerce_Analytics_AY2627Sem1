import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.splits import (day_blocked_time_series_folds, available_outcome_folds,
                        chronological_split, available_training_rows)
from src.labels import labels_as_of, regression_targets_as_of
from src.tuning import best_f1_threshold, out_of_fold_scores, build_classifiers, tune_classifier


class BoostingWeightTests(unittest.TestCase):
    def test_refit_recomputes_weight_and_rejects_single_class(self):
        X = pd.DataFrame({'x': np.arange(6, dtype=float)})
        models = build_classifiers(['x'], [])
        for name in ['XGBoost', 'LightGBM']:
            with self.subTest(model=name):
                model = models[name].set_params(classifier__n_estimators=2)
                model.fit(X, pd.Series([0, 0, 0, 0, 0, 1]))
                self.assertEqual(model.named_steps['classifier'].get_params()['scale_pos_weight'], 5.0)
                model.fit(X, pd.Series([0, 0, 0, 1, 1, 1]))
                self.assertEqual(model.named_steps['classifier'].get_params()['scale_pos_weight'], 1.0)
                with self.assertRaisesRegex(ValueError, 'both classes'):
                    model.fit(X, pd.Series([0] * 6))

    def test_cv_and_search_use_only_each_training_folds_labels(self):
        X = pd.DataFrame({'x': np.arange(18, dtype=float)})
        y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0])
        folds = [(np.arange(6), np.arange(6, 12)),
                 (np.arange(12), np.arange(12, 18))]
        models = build_classifiers(['x'], [])
        for name in ['XGBoost', 'LightGBM']:
            with self.subTest(model=name):
                model = models[name].set_params(classifier__n_estimators=2)
                result = cross_validate(model, X, y, cv=folds, scoring='average_precision',
                                        return_estimator=True, error_score='raise', n_jobs=1)
                for fitted, (tr, _) in zip(result['estimator'], folds):
                    expected = (y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum()
                    self.assertEqual(fitted.named_steps['classifier'].get_params()['scale_pos_weight'], expected)
                tuned, _, _ = tune_classifier(model, {'classifier__n_estimators': [2]},
                                               X, y, folds, n_iter=1, n_jobs=1)
                refitted = clone(tuned).fit(X, y)
                self.assertEqual(refitted.named_steps['classifier'].get_params()['scale_pos_weight'],
                                 (y == 0).sum() / (y == 1).sum())
                self.assertTrue(np.isfinite(out_of_fold_scores(tuned, X, y, folds)[6:]).all())


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
        targets = regression_targets_as_of(rows, '2018-04-01', 45)['target_as_of_run']
        folds = available_outcome_folds(
            rows, [(np.arange(2), np.array([2]))], targets,
            lambda r, d: regression_targets_as_of(r, d, 45), 'target_as_of_run')
        np.testing.assert_array_equal(folds[0][0], [0])
        self.assertEqual(regression_targets_as_of(rows.iloc[[0]], '2018-03-01', 45)
                         ['target_as_of_run'].iloc[0], 45)


class ThresholdTests(unittest.TestCase):
    def test_best_f1_threshold_separates_perfectly_ranked_scores(self):
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
        threshold, f1 = best_f1_threshold(y, scores)
        self.assertEqual(threshold, 0.7)
        self.assertEqual(f1, 1.0)

    def test_best_f1_threshold_ignores_missing_scores(self):
        threshold, f1 = best_f1_threshold([0, 1, 1], [np.nan, 0.4, 0.8])
        self.assertEqual(threshold, 0.4)
        self.assertEqual(f1, 1.0)


class OutOfFoldTests(unittest.TestCase):
    def test_rows_outside_validation_folds_stay_missing(self):
        dates = pd.Series(pd.date_range('2018-01-01', periods=40))
        X = pd.DataFrame({'x': np.arange(40, dtype=float)})
        y = pd.Series(np.arange(40) % 2)
        folds = day_blocked_time_series_folds(dates, n_splits=3)
        scores = out_of_fold_scores(LogisticRegression(), X, y, folds)
        validated = np.concatenate([valid for _, valid in folds])
        self.assertTrue(np.isnan(np.delete(scores, validated)).all())
        self.assertFalse(np.isnan(scores[validated]).any())


if __name__ == '__main__':
    unittest.main()
