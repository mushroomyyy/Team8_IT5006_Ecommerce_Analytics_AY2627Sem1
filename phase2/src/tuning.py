"""Classifier builders (Dummy, plain Logistic, Random Forest) and chronological-CV helpers."""
import warnings

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import RANDOM_STATE
from .linear_transforms import make_linear_preprocessor
from .preprocessing import make_preprocessor


def plain_logistic(random_state=RANDOM_STATE, ridge_fallback=False):
    """Unpenalised, unweighted logistic regression (lbfgs, max_iter=5000, tol=1e-8).

    The tight tolerance lets scikit-learn and statsmodels agree on identified coefficients.

    `ridge_fallback=True` uses a negligible L2 penalty (C=1e6), only for use if
    the plain fit fails to converge. scikit-learn 1.8+ replaced penalty=None by C=inf.
    """
    kwargs = dict(solver='lbfgs', max_iter=5000, tol=1e-8, random_state=random_state)
    if ridge_fallback:
        return LogisticRegression(C=1e6, **kwargs)
    if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 8):
        return LogisticRegression(C=np.inf, **kwargs)
    return LogisticRegression(penalty=None, **kwargs)


def fit_logistic_checked(pipeline, X, y):
    """Fit a pipeline ending in 'classifier'; on a convergence warning refit with C=1e6 L2.

    Returns (fitted pipeline, info) where info records whether the fallback was used.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = clone(pipeline).fit(X, y)
    if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return fitted, {'converged': True, 'fallback': None}
    fallback = clone(pipeline).set_params(classifier=plain_logistic(ridge_fallback=True))
    return fallback.fit(X, y), {'converged': False, 'fallback': 'L2 penalty with C=1e6'}


def build_classifiers(num_cols, cat_cols, random_state=RANDOM_STATE, linear_kwargs=None):
    """Return the Dummy (prior), plain Logistic and default Random Forest pipelines.

    The logistic preprocessing is the linear one (`linear_kwargs` turns on log1p,
    cyclic and squared terms); Random Forest uses the raw features.
    """
    linear = make_linear_preprocessor(num_cols, cat_cols, **(linear_kwargs or {}))
    raw = make_preprocessor(num_cols, cat_cols, reference_categories=True)
    return {
        'Dummy (prior)': DummyClassifier(strategy='prior'),
        'Logistic Regression': Pipeline([
            ('preprocessor', linear), ('classifier', plain_logistic(random_state))]),
        'Random Forest': Pipeline([
            ('preprocessor', raw),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=random_state,
                                                  n_jobs=-1))]),
    }


def tune_classifier(model, search_space, X, y, folds, n_iter, scoring='average_precision',
                    random_state=RANDOM_STATE, n_jobs=-1):
    """Random search over `search_space`; returns (unfitted tuned model, best params, search)."""
    search = RandomizedSearchCV(model, search_space, n_iter=n_iter, cv=folds, scoring=scoring,
                                random_state=random_state, n_jobs=n_jobs, refit=False)
    search.fit(X, y)
    return clone(model).set_params(**search.best_params_), search.best_params_, search


def out_of_fold_scores(model, X, y, folds):
    """Late-class scores for each validation row, fitted only on that fold's training rows.

    Rows that never fall in a validation fold (the earliest block) stay NaN.
    """
    scores = np.full(len(X), np.nan)
    for train_idx, valid_idx in folds:
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, list(fitted.classes_).index(1)]
    return scores


def best_f1_threshold(y_true, scores):
    """Threshold that maximises F1 on (y_true, scores); returns (threshold, f1)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    mask = ~np.isnan(scores)
    precision, recall, thresholds = precision_recall_curve(y_true[mask], scores[mask])
    f1 = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)
    best = int(np.nanargmax(f1[:-1]))  # The last point has no threshold
    return float(thresholds[best]), float(f1[best])


def params_to_json(params):
    """Convert numpy values in best_params_ to plain Python types for saving."""
    return {key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in params.items()}
