"""Model choice on Validation: day-level bootstrap against the leader, simplest tied model wins.

The rule is fixed before any result is seen. CV tunes hyperparameters and selects
features inside each model family; only Validation chooses between models, and June
never enters a choice.

1. The leader is the best Validation score (highest AP, lowest RMSE).
2. Whole approval days are resampled with replacement (orders on one day share
   carrier load and calendar effects), every model is rescored on the same resamples,
   and each model's difference from the leader gets a percentile interval.
3. A model whose interval includes 0 is tied with the leader.
4. Among tied models the simplest wins: linear, then single tree, then forest; within
   a family, fewer fitted parameters.

`paired_fold_differences` serves a different question (did tuning help?) and uses
the CV folds with the Nadeau-Bengio variance correction for overlapping training sets.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from . import RANDOM_STATE

FAMILY_RANK = {'linear': 0, 'tree': 1, 'forest': 2}


def average_precision(y, score):
    return average_precision_score(y, score)


def rmse(y, predicted):
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(predicted, dtype=float)) ** 2)))


METRICS = {'classification': (average_precision, True), 'regression': (rmse, False)}


def model_size(fitted, family):
    """Fitted parameter count: coefficients + intercept, leaves of a tree, or leaves over all trees."""
    estimator = fitted.steps[-1][1] if hasattr(fitted, 'steps') else fitted
    if family == 'linear':
        return int(np.size(estimator.coef_)) + int(np.size(getattr(estimator, 'intercept_', 0)))
    if family == 'tree':
        return int(estimator.get_n_leaves())
    if family == 'forest':
        return int(sum(tree.get_n_leaves() for tree in estimator.estimators_))
    raise ValueError(f'Unknown family {family!r}')


def bootstrap_vs_leader(y, predictions, days, task, n_boot=2000, level=0.95, random_state=RANDOM_STATE):
    """Each model's Validation score, its difference from the leader, and a day-level bootstrap interval.

    `predictions` maps a model code to scores aligned with `y`; `days` gives each row's
    approval day (the resampling unit). Pass `days=None` to resample single orders instead.
    Differences are model minus leader, so they are <= 0 for AP and >= 0 for RMSE.
    """
    metric, higher_is_better = METRICS[task]
    y = np.asarray(y, dtype=float)
    preds = {code: np.asarray(p, dtype=float) for code, p in predictions.items()}
    if any(len(p) != len(y) for p in preds.values()):
        raise ValueError('Every prediction must align with y')
    units = np.arange(len(y)) if days is None else pd.to_datetime(pd.Series(days)).dt.normalize().to_numpy()
    groups = [np.flatnonzero(units == u) for u in np.unique(units)]

    point = {code: metric(y, p) for code, p in preds.items()}
    leader = (max if higher_is_better else min)(point, key=point.get)
    rng = np.random.default_rng(random_state)
    draws = {code: np.empty(n_boot) for code in preds}
    for b in range(n_boot):
        index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for code, p in preds.items():
            draws[code][b] = metric(y[index], p[index])
    tail = 100 * (1 - level) / 2
    rows = []
    for code in preds:
        diff = draws[code] - draws[leader]
        low, high = np.percentile(diff, [tail, 100 - tail])
        rows.append({'model': code, 'validation': point[code], 'diff_vs_leader': point[code] - point[leader],
                     'ci_low': float(low), 'ci_high': float(high), 'leader': code == leader,
                     'tied': bool(code == leader or low <= 0 <= high)})
    table = pd.DataFrame(rows)
    table.attrs.update({'leader': leader, 'n_boot': n_boot, 'level': level,
                        'unit': 'order' if days is None else 'approval day', 'n_units': len(groups)})
    return table


def choose_model(table, families, sizes):
    """Add family, size and the decision to a `bootstrap_vs_leader` table; return (choice, table).

    `families` maps a code to 'linear', 'tree' or 'forest'; `sizes` maps a code to its
    fitted parameter count. Models without a family (baselines) are not selectable.
    """
    out = table.copy()
    out['family'] = out['model'].map(families)
    out['parameters'] = out['model'].map(sizes)
    eligible = out[out['tied'] & out['family'].notna()]
    if eligible.empty:
        raise ValueError('No selectable model is tied with the leader')
    key = lambda code: (FAMILY_RANK[families[code]], sizes[code])
    choice = min(eligible['model'], key=key)

    def reason(row):
        if pd.isna(row['family']):
            return 'baseline: not selectable'
        if row['model'] == choice:
            return 'selected: simplest model tied with the leader'
        if not row['tied']:
            return 'not tied: interval excludes 0'
        return 'tied, but a simpler tied model exists'

    out['selected'] = out['model'] == choice
    out['decision'] = out.apply(reason, axis=1)
    out.attrs.update(table.attrs)
    return choice, out


def paired_fold_differences(cv_folds, a, b, metric, folds, higher_is_better=True, level=0.95):
    """Fold-wise `a - b` for one metric with a Nadeau-Bengio corrected t-interval.

    Expanding folds share training rows, so the plain t-interval is too narrow; the
    corrected variance is (1/J + n_valid/n_train) x sample variance, with the ratio
    averaged over folds. `folds` are the (train_idx, valid_idx) pairs used for `cv_folds`.
    """
    pick = lambda code: cv_folds[cv_folds['model'] == code].sort_values('fold')[metric].to_numpy()
    diff = pick(a) - pick(b)
    j = len(diff)
    ratio = float(np.mean([len(valid) / len(train) for train, valid in folds]))
    variance = (1 / j + ratio) * diff.var(ddof=1)
    half = stats.t.ppf(1 - (1 - level) / 2, j - 1) * np.sqrt(variance)
    better = diff > 0 if higher_is_better else diff < 0
    return {'comparison': f'{a} - {b}', 'metric': metric, 'mean_diff': float(diff.mean()),
            'sd_diff': float(diff.std(ddof=1)), 'ci_low': float(diff.mean() - half),
            'ci_high': float(diff.mean() + half), 'folds_better': f'{int(better.sum())}/{j}',
            'fold_diffs': ', '.join(f'{d:+.4f}' for d in diff)}
