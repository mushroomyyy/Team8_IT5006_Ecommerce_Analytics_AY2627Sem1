"""Variable selection for the plain linear models, using Train rows only.

A unit is a raw feature. The three categoricals are single units (the whole dummy
block moves together) and transformed variants (log, sine/cosine) travel
with their raw feature.

Methods: forward stepwise scored by chronological CV with the 1-SE rule (primary),
forward stepwise by BIC / AIC on Train (cross-check), and a light Lasso check.
Every selection function takes the Train rows, the declared Train order IDs and,
for CV, the folds, and raises if given any row outside that Train set.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, get_scorer
from sklearn.pipeline import Pipeline

from . import RANDOM_STATE
from .linear_transforms import make_linear_preprocessor, readable_names
from .tuning import plain_logistic

log = logging.getLogger(__name__)
INTERCEPT = '(intercept only)'


def check_train_only(train_rows, train_ids, folds=None, id_col='order_id'):
    """Raise unless every row is in the declared Train set and the folds index only these rows."""
    ids = train_rows[id_col] if id_col in train_rows.columns else train_rows.index.to_series()
    outside = set(ids) - set(train_ids)
    if outside:
        raise ValueError(f'{len(outside):,} rows are outside the declared Train set, '
                         f'for example {list(outside)[:3]}')
    for k, (train_idx, valid_idx) in enumerate(folds or [], 1):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(f'Fold {k} is empty')
        if max(train_idx.max(), valid_idx.max()) >= len(train_rows):
            raise ValueError(f'Fold {k} refers to rows beyond the Train rows supplied')
        if np.intersect1d(train_idx, valid_idx).size:
            raise ValueError(f'Fold {k} trains and validates on the same rows')


def make_unit_pipeline_builder(task, num_cols, cat_cols, **transform_kwargs):
    """Return `build(units)`: an unfitted plain linear/logistic pipeline for a subset of units.

    `transform_kwargs` are the `make_linear_preprocessor` options (log_cols, cyclic,
    merge_mixed_route). Columns keep their original order whatever order units entered.
    """
    def build(units):
        chosen = set(units)
        num = [c for c in num_cols if c in chosen]
        cat = [c for c in cat_cols if c in chosen]
        pre = make_linear_preprocessor(num, cat, **transform_kwargs)
        if task == 'classification':
            return Pipeline([('preprocessor', pre), ('classifier', plain_logistic())])
        return Pipeline([('preprocessor', pre), ('model', LinearRegression())])
    return build


def intercept_model(y):
    """Constant model for the empty step: class prior for binary y, else the mean."""
    if pd.Series(y).nunique() <= 2:
        return DummyClassifier(strategy='prior')
    return DummyRegressor(strategy='mean')


def _evaluate_candidate(unit, units, X, y, folds, pipeline, scoring):
    """CV score of one candidate pipeline: per-fold scores, encoded width, convergence count."""
    scorer = get_scorer(scoring)
    scores, warned = [], 0
    for train_idx, valid_idx in folds:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(float(scorer(model, X.iloc[valid_idx], y.iloc[valid_idx])))
        warned += any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if units:
        width = len(readable_names(clone(pipeline.named_steps['preprocessor']).fit(X)))
    else:
        width = 0
    scores = np.array(scores)
    return {'unit': unit, 'n_units': len(units), 'n_encoded_columns': width,
            'cv_mean': scores.mean(), 'cv_se': scores.std(ddof=1) / np.sqrt(len(scores)),
            **{f'fold_{k}': s for k, s in enumerate(scores, 1)}, 'n_convergence_warnings': warned}


def forward_stepwise_cv(train_rows, y, units, build_pipeline, folds, scoring, train_ids,
                        intercept=None, n_jobs=-1, early_stop=None):
    """Forward selection judged by chronological CV; returns (path, candidates).

    Starts from an intercept-only model (Dummy prior / mean) and adds, at each step,
    the unit with the best mean CV score, until all units are in. `scoring` is a
    scikit-learn scorer name (higher is better, so RMSE appears as negative RMSE).
    `build_pipeline(units)` returns an unfitted pipeline for that unit subset.
    `early_stop=k` stops after k consecutive steps that do not beat the best mean so
    far; the stop is logged and stored in `path.attrs['stopped_early_after_step']`.

    path: step, added_unit, n_units, n_encoded_columns, cv_mean, cv_se.
    candidates: every unit tried at every step, with fold scores and `chosen`.
    """
    check_train_only(train_rows, train_ids, folds)
    units = list(units)
    X, y = train_rows[units].reset_index(drop=True), pd.Series(np.asarray(y))
    start = _evaluate_candidate(INTERCEPT, [], X, y, folds,
                                intercept if intercept is not None else intercept_model(y),
                                scoring)
    path = [{'step': 0, 'added_unit': INTERCEPT, **{k: start[k] for k in
             ['n_units', 'n_encoded_columns', 'cv_mean', 'cv_se']}}]
    candidates, selected, remaining = [], [], units.copy()
    best_mean, stale, stopped = start['cv_mean'], 0, None
    for step in range(1, len(units) + 1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_candidate)(
                unit, selected + [unit], X[selected + [unit]], y, folds,
                build_pipeline(selected + [unit]), scoring)
            for unit in remaining)
        means = np.array([np.nan_to_num(r['cv_mean'], nan=-np.inf) for r in results])
        best = int(np.argmax(means))  # First of equals: deterministic in `units` order
        for i, r in enumerate(results):
            candidates.append({'step': step, **r, 'chosen': i == best})
        chosen = results[best]
        selected.append(chosen['unit'])
        remaining.remove(chosen['unit'])
        path.append({'step': step, 'added_unit': chosen['unit'], **{k: chosen[k] for k in
                     ['n_units', 'n_encoded_columns', 'cv_mean', 'cv_se']}})
        if chosen['cv_mean'] > best_mean:
            best_mean, stale = chosen['cv_mean'], 0
        else:
            stale += 1
        if early_stop and stale >= early_stop and remaining:
            stopped = step
            log.warning('Early stop after step %d: %d consecutive steps without a new best CV mean',
                        step, stale)
            break
    path, candidates = pd.DataFrame(path), pd.DataFrame(candidates)
    path.attrs['stopped_early_after_step'] = stopped
    return path, candidates


def one_se_rule(path, higher_is_better=True, min_step=1):
    """Smallest step whose mean CV score is within 1 SE of the best step; returns (chosen, best).

    The SE is that of the best step. Steps before `min_step` (the intercept-only
    model) are not eligible.
    """
    eligible = path[path['step'] >= min_step]
    sign = 1 if higher_is_better else -1
    best = eligible.loc[(sign * eligible['cv_mean']).idxmax()]
    cutoff = sign * best['cv_mean'] - best['cv_se']
    qualifying = eligible[sign * eligible['cv_mean'] >= cutoff]
    return int(qualifying['step'].min()), int(best['step'])


def units_through_step(path, step):
    """Units added up to and including `step`, in the order they entered."""
    return path.loc[(path['step'] >= 1) & (path['step'] <= step), 'added_unit'].tolist()


def _fit_ic(task, y, exog, fallback_method=None):
    """Fit OLS / Logit; returns (aic, bic, method) or a failure reason string."""
    if task == 'regression':
        result = sm.OLS(y, exog).fit()
        return result.aic, result.bic, 'ols'
    for method in ['newton'] + ([fallback_method] if fallback_method else []):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = sm.Logit(y, exog).fit(disp=0, maxiter=200 if method == 'newton' else 1000,
                                               method=method)
        except Exception as error:  # Perfect separation, singular matrix, ...
            reason = f'{type(error).__name__}: {error}'
            continue
        if result.mle_retvals.get('converged', True):
            return result.aic, result.bic, method
        reason = 'did not converge'
    return reason


def forward_stepwise_ic(train_rows, y, units, design_builder, task, train_ids,
                        criterion='bic', n_jobs=-1, fallback_method='bfgs'):
    """Forward selection by in-sample AIC or BIC on Train; returns (path, selected_units).

    `design_builder(train_rows)` returns (design, unit_columns): the standardised,
    reference-coded design fitted on Train and the unit -> encoded columns mapping
    (see `linear_transforms.linear_design`). Fits statsmodels OLS or unweighted Logit
    with an intercept; a unit costs all of its encoded columns. Stops when no addition
    lowers the criterion. Candidates whose Logit fails to converge are skipped and
    logged; they are in `path.attrs['skipped']`. Quasi-separation (a rare level with no
    late orders, e.g. route_type '3. Mixed') makes Newton fail for such candidates, so by
    default those fits are retried with BFGS and only skipped if that fails too (the
    fitting method is in the `method` column of `path`). Pass `fallback_method=None`
    to skip every Newton failure.
    AIC and BIC are statsmodels' (-2 lnL + 2k and -2 lnL + k ln n, k counting the
    intercept), so only the constant offset differs from using encoded columns alone.

    path: step, added_unit, k (encoded columns), aic, bic, method.
    """
    if criterion not in ('aic', 'bic'):
        raise ValueError("criterion must be 'aic' or 'bic'")
    check_train_only(train_rows, train_ids)
    design, unit_cols = design_builder(train_rows)
    missing = [u for u in units if u not in unit_cols]
    if missing:
        raise ValueError(f'Units missing from the design: {missing}')
    y = np.asarray(y, dtype=float)

    def fit(cols):
        exog = (np.ones((len(design), 1)) if not cols
                else sm.add_constant(design[cols].to_numpy(float), prepend=True, has_constant='add'))
        return _fit_ic(task, y, exog, fallback_method)

    base = fit([])
    path = [{'step': 0, 'added_unit': INTERCEPT, 'k': 0, 'aic': base[0], 'bic': base[1],
             'method': base[2]}]
    current = base[0 if criterion == 'aic' else 1]
    selected, columns, skipped = [], [], []
    remaining = list(units)
    column = 0 if criterion == 'aic' else 1
    while remaining:
        step = len(selected) + 1
        fits = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(fit)(columns + unit_cols[u]) for u in remaining)
        scored = []
        for unit, result in zip(remaining, fits):
            if isinstance(result, str):
                skipped.append({'step': step, 'unit': unit, 'reason': result})
                log.warning('Step %d: skipped %s (%s)', step, unit, result)
            else:
                scored.append((result[column], unit, result))
        if not scored:
            break
        value, unit, result = min(scored, key=lambda item: item[0])  # First of equals wins
        if value >= current:
            break
        selected.append(unit)
        remaining.remove(unit)
        columns = columns + unit_cols[unit]
        current = value
        path.append({'step': step, 'added_unit': unit, 'k': len(columns),
                     'aic': result[0], 'bic': result[1], 'method': result[2]})
    path = pd.DataFrame(path)
    path.attrs['skipped'] = pd.DataFrame(skipped, columns=['step', 'unit', 'reason'])
    return path, selected


def _l1_logistic(C, random_state):
    """L1 logistic regression; scikit-learn 1.8+ spells penalty='l1' as l1_ratio=1."""
    kwargs = dict(C=C, solver='saga', max_iter=2000, tol=1e-3, random_state=random_state)
    if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **kwargs)
    return LogisticRegression(penalty='l1', **kwargs)


def _l1_fold_ap(C, X, y, train_idx, valid_idx, random_state):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = _l1_logistic(C, random_state).fit(X.iloc[train_idx], y.iloc[train_idx])
    return average_precision_score(y.iloc[valid_idx], model.predict_proba(X.iloc[valid_idx])[:, 1])


def lasso_units(train_rows, y, units, design_builder, task, folds, train_ids,
                n_jobs=-1, random_state=RANDOM_STATE):
    """Lasso / L1-logistic cross-check; returns a dict with the selected units.

    Uses the same design and folds. Regression: LassoCV with its default alpha grid.
    Classification: L1 logistic over LogisticRegressionCV's default grid
    (10 values of C from 1e-4 to 1e4), chosen by mean CV average precision.
    A categorical block is kept if any of its dummies is non-zero.
    Returns {'selected_units', 'penalty', 'coef'} with `penalty` = alpha or C.
    """
    check_train_only(train_rows, train_ids, folds)
    design, unit_cols = design_builder(train_rows)
    design = design.reset_index(drop=True)
    y = pd.Series(np.asarray(y))
    if task == 'regression':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = LassoCV(cv=folds, random_state=random_state, max_iter=5000,
                            n_jobs=n_jobs).fit(design, y)
        coef, penalty = model.coef_, float(model.alpha_)
    else:
        grid = np.logspace(-4, 4, 10)
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_l1_fold_ap)(C, design, y, tr, va, random_state)
            for C in grid for tr, va in folds)
        mean_ap = np.asarray(scores).reshape(len(grid), len(folds)).mean(axis=1)
        penalty = float(grid[int(np.argmax(mean_ap))])  # First of equals: strongest penalty
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = _l1_logistic(penalty, random_state).fit(design, y)
        coef = model.coef_[0]
    coef = pd.Series(coef, index=design.columns)
    selected = [u for u in units if (coef[unit_cols[u]].abs() > 0).any()]
    return {'selected_units': selected, 'penalty': penalty, 'coef': coef}


def selection_overlap(units, cv_path, cv_step, bic_selected, aic_selected, lasso_selected):
    """Unit x method table: CV-stepwise (and its entry step), BIC, AIC, Lasso, and a count."""
    entered = cv_path.set_index('added_unit')['step']
    chosen = set(units_through_step(cv_path, cv_step))
    table = pd.DataFrame({'unit': list(units)})
    table['cv_stepwise'] = table['unit'].isin(chosen)
    table['cv_step_entered'] = table['unit'].map(entered).astype('Int64')
    table['bic'] = table['unit'].isin(set(bic_selected))
    table['aic'] = table['unit'].isin(set(aic_selected))
    table['lasso'] = table['unit'].isin(set(lasso_selected))
    table['n_methods'] = table[['cv_stepwise', 'bic', 'aic', 'lasso']].sum(axis=1)
    return table.sort_values(['n_methods', 'cv_step_entered'], ascending=[False, True],
                             kind='stable', ignore_index=True)
