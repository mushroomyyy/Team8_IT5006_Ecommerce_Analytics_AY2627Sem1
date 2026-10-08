"""Regression models: Olist promise baseline, waiting-period clipping and pipeline builders."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from . import LOOKBACK, RANDOM_STATE
from .linear_transforms import make_linear_preprocessor
from .preprocessing import make_preprocessor

WAITING_PERIOD = LOOKBACK


def clip_to_waiting_period(predicted, promised_lead_days, waiting_period=WAITING_PERIOD):
    """Limit predicted days from the promise to a lead time between 0 and `waiting_period` days."""
    promised = np.asarray(promised_lead_days, dtype=float)
    return np.clip(predicted, -promised, waiting_period - promised)


class PromiseBaseline(DummyRegressor):
    """Predicts delivery on the promised date: zero days off (lead time capped at the waiting period)."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y)

    def predict(self, X):
        return clip_to_waiting_period(np.zeros(len(X)), X['promised_lead_days'])


def build_regressors(num_cols, cat_cols, random_state=RANDOM_STATE, linear_kwargs=None):
    """Return the Dummy (median), Olist promise, OLS and default Random Forest models.

    OLS uses the linear preprocessor (`linear_kwargs` turns on log1p, cyclic and
    squared terms); Random Forest uses the raw features. RF n_jobs=1 so parallelism
    happens across CV folds and search candidates.
    """
    linear = make_linear_preprocessor(num_cols, cat_cols, **(linear_kwargs or {}))
    raw = make_preprocessor(num_cols, cat_cols, reference_categories=True)
    return {
        'Dummy (median)': DummyRegressor(strategy='median'),
        'Olist promise': PromiseBaseline(),
        'OLS': Pipeline([('preprocessor', linear), ('model', LinearRegression())]),
        'Random Forest': Pipeline([
            ('preprocessor', raw),
            ('model', RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=1))]),
    }


def score_cohort(fitted, cohort, feature_cols):
    """Predicted days from the promise for each order, clipped to the waiting period."""
    return clip_to_waiting_period(fitted.predict(cohort[feature_cols]), cohort['promised_lead_days'])
