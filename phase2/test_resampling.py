import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import cv_classification_metrics
from src.resampling import STRATEGIES, binary_columns, paired_gain, with_resampling

NUM, CAT = ['a', 'b', 'flag'], ['state']


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n),
                      'flag': rng.integers(0, 2, n).astype(float),
                      'state': rng.choice(['SP', 'RJ', 'MG'], n)})
    X.loc[::37, 'a'] = np.nan
    y = pd.Series((rng.random(n) < 0.08 + 0.1 * (X['b'] > 1)).astype(int))
    return X, y


class RecordingModel(ClassifierMixin, BaseEstimator):
    """Remembers how many rows it was fitted on and how many it predicted."""
    fit_sizes, predict_sizes = [], []

    def fit(self, X, y):
        RecordingModel.fit_sizes.append(len(X))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        RecordingModel.predict_sizes.append(len(X))
        return np.tile([0.9, 0.1], (len(X), 1))


def base_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([('n', Pipeline([('i', SimpleImputer()), ('s', StandardScaler())]), ['a', 'b', 'flag']),
                             ('c', OneHotEncoder(handle_unknown='ignore'), ['state'])])
    return Pipeline([('preprocessor', pre), ('classifier', LogisticRegression(max_iter=500))])


class ResamplingTests(unittest.TestCase):
    def test_binary_columns_found(self):
        X, _ = toy()
        self.assertEqual(binary_columns(X, NUM), ['flag'])

    def test_each_strategy_fits_and_predicts_all_rows(self):
        X, y = toy()
        for strategy in STRATEGIES:
            model = with_resampling(base_model(), strategy, NUM, CAT, ['flag']).fit(X, y)
            self.assertEqual(model.predict_proba(X).shape, (len(X), 2), strategy)

    def test_none_leaves_model_unchanged(self):
        self.assertIsInstance(with_resampling(base_model(), 'none', NUM, CAT), Pipeline)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            with_resampling(base_model(), 'bogus', NUM, CAT)

    def test_resampling_only_changes_training_rows(self):
        X, y = toy()
        RecordingModel.fit_sizes, RecordingModel.predict_sizes = [], []
        record = Pipeline([('classifier', RecordingModel())])
        folds = [(np.arange(0, 400), np.arange(430, 600))]
        for strategy in ['undersample', 'oversample', 'smotenc']:
            model = with_resampling(record, strategy, NUM, CAT, ['flag'])
            model.set_params(model__classifier=RecordingModel())
            cv_classification_metrics(model, X.assign(state=X['state']), y, folds)
        n_late = int(y.iloc[:400].sum())
        n_ontime = 400 - n_late
        self.assertEqual(RecordingModel.fit_sizes, [2 * n_late, 2 * n_ontime, 2 * n_ontime])
        self.assertEqual(RecordingModel.predict_sizes, [170, 170, 170])

    def test_balanced_after_resampling(self):
        X, y = toy()
        model = with_resampling(base_model(), 'smotenc', NUM, CAT, ['flag'])
        Xs, ys = model.named_steps['sampler'].fit_resample(
            model.named_steps['impute'].fit_transform(X), y)
        self.assertEqual(ys.value_counts().nunique(), 1)
        self.assertEqual(len(Xs), len(ys))

    def test_paired_gain(self):
        table = pd.DataFrame({'model': 'M', 'strategy': ['none'] * 3 + ['x'] * 3,
                              'fold': [1, 2, 3] * 2, 'avg_precision': [.1, .2, .3, .2, .3, .4]})
        mean, se = paired_gain(table, 'M', 'x')
        self.assertAlmostEqual(mean, 0.1)
        self.assertAlmostEqual(se, 0.0)


if __name__ == '__main__':
    unittest.main()
