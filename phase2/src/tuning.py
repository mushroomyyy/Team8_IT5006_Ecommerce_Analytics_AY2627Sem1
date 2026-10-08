"""Plain (unpenalised, unweighted) logistic regression and a convergence-checked fit."""
import warnings

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from . import RANDOM_STATE


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
