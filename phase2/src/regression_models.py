"""Regression models: Olist promise baseline, waiting-period clipping and pipeline builders."""
import numpy as np
import pandas as pd
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


def default_rf_pipeline(num_cols, cat_cols, random_state=RANDOM_STATE):
    """Untuned Random Forest (scikit-learn defaults) on the raw features: the R3d baseline."""
    return Pipeline([
        ('preprocessor', make_preprocessor(num_cols, cat_cols, reference_categories=True)),
        ('model', RandomForestRegressor(random_state=random_state, n_jobs=-1))])


class ClippedDaysScorer:
    """Adapter that lets `run_inference` score a regressor day by day.

    `run_inference` reads `classes_` and `predict_proba`; this wrapper returns the
    clipped predicted days as the second ("positive") column. Rename the output
    column to `predicted_days` afterwards.
    """

    classes_ = np.array([0, 1])

    def __init__(self, fitted):
        self.fitted = fitted

    def predict_proba(self, X):
        days = score_cohort(self.fitted, X, list(X.columns))
        return np.column_stack([np.zeros(len(days)), days])


def heteroscedasticity_table(ols_result, n_groups=5):
    """Breusch-Pagan test plus residual SD by fitted-value group for an OLS fit.

    Returns (summary, groups): a one-row frame with the test statistic and p-value, and
    the residual SD and mean in `n_groups` equal-count groups of fitted values.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    residual, fitted = np.asarray(ols_result.resid), np.asarray(ols_result.fittedvalues)
    statistic, p_value, _, _ = het_breuschpagan(residual, ols_result.model.exog)
    summary = pd.DataFrame([{'Breusch-Pagan LM': statistic, 'p-value': p_value}])
    group = pd.qcut(fitted, n_groups, labels=False, duplicates='drop') + 1
    groups = (pd.DataFrame({'Fitted-value group': group, 'fitted': fitted, 'residual': residual})
              .groupby('Fitted-value group').agg(mean_fitted=('fitted', 'mean'),
                                                 residual_mean=('residual', 'mean'),
                                                 residual_sd=('residual', 'std'),
                                                 orders=('residual', 'size')).reset_index())
    return summary, groups
