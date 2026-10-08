"""Metrics and coverage tables for the classification and regression tracks."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    mean_absolute_error, mean_squared_error, median_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_validate

from .inference import run_inference


def classification_metrics(y_true, y_pred, y_prob=None):
    """Imbalance-aware classification metrics; the late class (1) is positive."""
    two_classes = pd.Series(y_true).nunique() == 2
    return {
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None and two_classes else np.nan,
        'avg_precision': average_precision_score(y_true, y_prob) if y_prob is not None and two_classes else np.nan,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_pipeline(pipeline, X_test, y_test, threshold=0.5):
    """Return (metrics, y_pred, y_prob) for a fitted classification pipeline on raw features."""
    y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    y_pred = (y_prob >= threshold).astype(int)
    return classification_metrics(y_test, y_pred, y_prob), y_pred, y_prob


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R², plus median absolute error and mean bias (pred - actual)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'bias': float(np.mean(y_pred - y_true)),
    }


def get_coverage(prob, y_true):
    """Precision and recall (coverage) by predicted-risk decile; decile 1 is riskiest."""
    predictions_df = pd.DataFrame({'actual_late_delivery': y_true,
                                   'prob_late_delivery': prob}, index=y_true.index)
    bins = pd.qcut(predictions_df['prob_late_delivery'], q=10, labels=False, duplicates='drop')
    # Identical scores form one group rather than silently dropping all rows
    predictions_df['decile'] = (pd.Series(1, index=predictions_df.index, dtype='Int64')
                                if bins.isna().all() else (bins.max() - bins + 1).astype('Int64'))
    summary = predictions_df.groupby('decile').agg(
        number_of_orders=('actual_late_delivery', 'size'),
        positive_labels=('actual_late_delivery', 'sum'),
    ).sort_index()
    summary['cumulative_orders'] = summary['number_of_orders'].cumsum()
    summary['cumulative_positive_labels'] = summary['positive_labels'].cumsum()
    summary['late_delivery_rate'] = summary['positive_labels'] / summary['number_of_orders']
    summary['cumulative_late_delivery_rate'] = (summary['cumulative_positive_labels']
                                                / summary['cumulative_orders'])
    total_positive = summary['positive_labels'].sum()
    summary['coverage'] = (summary['cumulative_positive_labels'] / total_positive
                           if total_positive > 0 else float('nan'))
    return summary.reset_index()


def risk_decile_coverage(predictions):
    """Capture and lift for equally sized groups, highest predicted risk first.

    Unknown outcomes stay in the scored population (operational capacity), but
    only known late outcomes contribute to capture. Lift follows the notebook's
    top-10% definition: share of all known late orders / share of scored orders.
    """
    ranked = predictions.sort_values(
        ['predicted_probability', 'order_id'],
        ascending=[False, True], kind='stable',
    ).reset_index(drop=True)
    if len(ranked) < 10:
        raise ValueError('At least 10 scored orders are required for ten deciles.')

    known_labels = ranked['actual_label'].dropna()
    if not known_labels.isin([0, 1]).all():
        raise ValueError('Known classification labels must be 0 or 1.')

    total_late = int(known_labels.sum())
    late_rate = total_late / len(ranked)
    rows = []
    start = 0
    cumulative_late = 0

    for decile in range(1, 11):
        stop = int(np.ceil(len(ranked) * decile / 10))
        group = ranked.iloc[start:stop]
        late = int(group['actual_label'].sum())
        cumulative_late += late

        rows.append({
            'decile': decile,
            'scored_orders': len(group),
            'known_orders': int(group['actual_label'].notna().sum()),
            'late_orders': late,
            'capture_pct': 100 * late / total_late if total_late else np.nan,
            'lift': (late / len(group)) / late_rate if late_rate > 0 else np.nan,
            'cumulative_capture_pct': (
                100 * cumulative_late / total_late if total_late else np.nan
            ),
        })
        start = stop

    return pd.DataFrame(rows)


def cv_summary(model, X, y, folds, scoring, n_jobs=-1):
    """Mean and sample SD (ddof=1) of each CV score across the folds.

    `scoring` maps metric name to a scikit-learn scorer. Scorers named `neg_*`
    are flipped back to natural units (e.g. RMSE in days).
    """
    scores = cross_validate(model, X, y, cv=folds, scoring=scoring, n_jobs=n_jobs,
                            error_score='raise')
    row = {}
    for metric, scorer in scoring.items():
        values = scores[f'test_{metric}']
        if isinstance(scorer, str) and scorer.startswith('neg_'):
            values = -values
        row[f'cv_{metric}_mean'], row[f'cv_{metric}_std'] = values.mean(), values.std(ddof=1)
    return row


def classification_eval_metrics(y_true, y_prob, top_frac=0.10, order_ids=None):
    """AP, ROC-AUC, Brier, and precision/recall/F1 at the top-10% cut-off and at 0.5.

    The top-`top_frac` group is the highest-scored share of ALL scored orders
    (operational capacity); unknown outcomes (NA) stay in it but never count as late.
    Capture is the share of known late orders in the group; lift is capture divided by
    the group's share of scored orders. Ties are broken by order_id when supplied.
    """
    y = pd.Series(y_true).reset_index(drop=True)
    prob = np.asarray(y_prob, dtype=float)
    known = y.notna().to_numpy()
    late = y.fillna(0).astype(int).to_numpy()
    n = len(prob)
    ids = np.arange(n) if order_ids is None else np.asarray(order_ids)
    order = np.lexsort((ids, -prob))
    top_count = max(1, int(np.ceil(n * top_frac)))
    in_top = np.zeros(n, dtype=bool)
    in_top[order[:top_count]] = True
    total_late = int(late[known].sum())
    top_late = int(late[in_top & known].sum())
    precision_top = top_late / top_count
    recall_top = top_late / total_late if total_late else np.nan
    f1_top = (2 * precision_top * recall_top / (precision_top + recall_top)
              if total_late and (precision_top + recall_top) else 0.0)
    yk, pk = late[known], prob[known]
    two_classes = len(np.unique(yk)) == 2
    flag = pk >= 0.5
    return {
        'avg_precision': average_precision_score(yk, pk) if two_classes else np.nan,
        'roc_auc': roc_auc_score(yk, pk) if two_classes else np.nan,
        'brier': brier_score_loss(yk, pk),
        'precision_top10': precision_top, 'recall_top10': recall_top, 'f1_top10': f1_top,
        'precision_at_0_5': precision_score(yk, flag, zero_division=0),
        'recall_at_0_5': recall_score(yk, flag, zero_division=0),
        'f1_at_0_5': f1_score(yk, flag, zero_division=0),
        'top10_capture': recall_top,
        'top10_lift': recall_top / (top_count / n) if total_late else np.nan,
        'late_rate': total_late / known.sum(), 'n_scored': n, 'n_known': int(known.sum()),
    }


def regression_eval_metrics(y_true, y_pred, late_label=None):
    """RMSE, MAE, R², median AE, bias, errors split by actual late/on-time, and the late flag.

    The derived late flag is prediction > 0 days. Actual late is `late_label` when
    given (NA rows skipped), else target > 0.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    if late_label is None:
        actual, usable = y_true > 0, np.ones(len(y_true), dtype=bool)
    else:
        label = pd.Series(late_label).reset_index(drop=True)
        usable, actual = label.notna().to_numpy(), label.fillna(0).astype(int).to_numpy().astype(bool)
    for name, mask in [('late', actual), ('on_time', ~actual)]:
        mask = mask & usable
        error = y_pred[mask] - y_true[mask]
        metrics[f'rmse_{name}'] = float(np.sqrt(np.mean(error ** 2))) if mask.any() else np.nan
        metrics[f'mae_{name}'] = float(np.mean(np.abs(error))) if mask.any() else np.nan
    flag = (y_pred > 0)[usable]
    metrics.update({
        'late_flag_precision': precision_score(actual[usable], flag, zero_division=0),
        'late_flag_recall': recall_score(actual[usable], flag, zero_division=0),
        'late_flag_f1': f1_score(actual[usable], flag, zero_division=0),
        'n_scored': len(y_true), 'n_actual_late': int((actual & usable).sum()),
    })
    return metrics


def _hash_state(digest, obj, seen):
    """Feed a stable description of a fitted estimator's state into `digest`."""
    if isinstance(obj, (type(None), bool, int, float, str, bytes, complex)):
        digest.update(repr((type(obj).__name__, obj)).encode())
    elif isinstance(obj, np.generic):
        _hash_state(digest, obj.item(), seen)
    elif isinstance(obj, np.ndarray):
        digest.update(f'nd{obj.dtype}{obj.shape}'.encode())
        if obj.dtype.names:  # Structured arrays (tree nodes): skip padding bytes
            for name in obj.dtype.names:
                _hash_state(digest, obj[name], seen)
        elif obj.dtype == object:
            _hash_state(digest, obj.tolist(), seen)
        else:
            digest.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (list, tuple)):
        digest.update(f'{type(obj).__name__}{len(obj)}'.encode())
        for item in obj:
            _hash_state(digest, item, seen)
    elif isinstance(obj, dict):
        digest.update(f'dict{len(obj)}'.encode())
        for key in sorted(obj, key=repr):
            _hash_state(digest, key, seen)
            _hash_state(digest, obj[key], seen)
    elif isinstance(obj, (set, frozenset)):
        _hash_state(digest, sorted(obj, key=repr), seen)
    elif isinstance(obj, (pd.Series, pd.Index)):
        _hash_state(digest, obj.to_numpy(), seen)
    elif callable(obj) and not isinstance(obj, BaseEstimator) and hasattr(obj, '__qualname__'):
        digest.update(f'{obj.__module__}.{obj.__qualname__}'.encode())
    elif id(obj) in seen:
        digest.update(b'<cycle>')
    else:
        seen = seen | {id(obj)}
        digest.update(type(obj).__qualname__.encode())
        if hasattr(obj, '__dict__'):
            state = vars(obj)
        elif hasattr(obj, '__getstate__'):  # Cython objects such as tree_.Tree
            state = obj.__getstate__()
        else:
            state = repr(obj)
        _hash_state(digest, state, seen)


def fingerprint(fitted):
    """Stable SHA-256 of a fitted model's parameters and learned state (pipelines included).

    Equal fingerprints before and after scoring show the model was not refitted.
    """
    digest = hashlib.sha256()
    _hash_state(digest, fitted, frozenset())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Shared result tables. `results` is a long frame with one row per (model, split)
# and one column per metric; `cv_folds` has one row per (model, fold). Both tasks use
# the same layout, so the regression notebook reuses these helpers unchanged.
# ---------------------------------------------------------------------------
CLASSIFICATION_CV_METRICS = ['avg_precision', 'roc_auc', 'brier', 'precision_top10', 'top10_capture',
                             'top10_lift']
SPLIT_ORDER = ['Train', 'CV', 'Validation', 'June', 'July', 'August']


def cv_classification_metrics(model, X, y, folds, metrics=None):
    """Fit `model` on each chronological training fold and score its validation fold.

    Returns a DataFrame with one row per fold (metrics from `classification_eval_metrics`).
    Resampling pipelines resample only the training part of each fold, because the
    validation fold only goes through `predict_proba`.
    """
    from sklearn.base import clone

    metrics = metrics or CLASSIFICATION_CV_METRICS
    rows = []
    for number, (train_idx, valid_idx) in enumerate(folds, 1):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = fitted.predict_proba(X.iloc[valid_idx])[:, list(fitted.classes_).index(1)]
        scores = classification_eval_metrics(y.iloc[valid_idx], prob)
        rows.append({'fold': number, **{m: scores[m] for m in metrics}})
    return pd.DataFrame(rows)


def cv_mean_sd(cv_folds):
    """Mean and sample SD (ddof=1) of every metric per model from a long per-fold frame."""
    grouped = cv_folds.drop(columns='fold').groupby('model', sort=False)
    return grouped.mean(), grouped.std(ddof=1)


def greedy_transform_cv(build_model, groups, X, y, folds, scoring, higher_is_better=True, initial=()):
    """Add transform groups one at a time and keep each only if mean CV score improves.

    `build_model(kept_groups)` returns an unfitted pipeline using those groups;
    `groups` is the ordered list of group names to try. Each group is also scored
    alone on the starting design (`initial` groups already kept). Returns (table, kept
    groups). `scoring` is a scorer name; `cv_mean` is in that scorer's own units.
    """
    sign = 1 if higher_is_better else -1

    def score(kept):
        values = cross_validate(build_model(kept), X, y, cv=folds, scoring=scoring,
                                n_jobs=-1, error_score='raise')['test_score']
        return sign * values.mean(), values.std(ddof=1)

    start = list(initial)
    base_mean, base_sd = score(start)
    rows = [{'design': '+'.join(start) or 'raw', 'groups': '+'.join(start), 'cv_mean': sign * base_mean, 'cv_sd': base_sd,
             'change_vs_kept': 0.0, 'kept': True}]
    kept, best = start, base_mean
    for group in groups:
        alone_mean, alone_sd = score(start + [group])
        mean, sd = score(kept + [group])
        improved = mean > best
        rows.append({'design': f'{rows[0]["design"]} + {group}', 'groups': '+'.join(start + [group]), 'cv_mean': sign * alone_mean,
                     'cv_sd': alone_sd, 'change_vs_kept': np.nan, 'kept': False})
        rows.append({'design': f'kept ({"+".join(kept) or "raw"}) + {group}', 'groups': '+'.join(kept + [group]),
                     'cv_mean': sign * mean, 'cv_sd': sd,
                     'change_vs_kept': sign * (mean - best), 'kept': improved})
        if improved:
            kept, best = kept + [group], mean
    return pd.DataFrame(rows), kept


def _tidy_split(split):
    return 'CV' if split.startswith('CV') else split


def comparison_table(results, cv_folds, metric, percent=False, splits=None):
    """Wide table (model x split) for one metric; CV shows mean +/- SD.

    `percent=True` multiplies by 100. Splits missing from `results` are skipped.
    """
    scale = 100 if percent else 1
    digits = 2 if percent else 3
    mean, sd = cv_mean_sd(cv_folds)
    wide = results.pivot(index='model', columns='split', values=metric) * scale
    table = pd.DataFrame(index=wide.index)
    for split in splits or SPLIT_ORDER:
        if split == 'CV':
            table['CV mean +/- SD'] = [f'{mean.loc[m, metric] * scale:.{digits}f} +/- '
                                       f'{sd.loc[m, metric] * scale:.{digits}f}' for m in wide.index]
        elif split in wide.columns:
            table[split] = wide[split].map(lambda v: f'{v:.{digits}f}')
    table.index.name = 'Model'
    return table.reset_index()


def overfitting_gaps(results, cv_folds, metrics):
    """Train (in-sample) minus CV mean for each model and metric (positive = higher on Train)."""
    mean, _ = cv_mean_sd(cv_folds)
    train = results[results['split'] == 'Train'].set_index('model')[metrics]
    return (train - mean[metrics].reindex(train.index)).rename_axis('model').reset_index()


def tuning_gain_table(results, cv_folds, tuned, default, metrics, splits=('CV', 'Validation', 'June')):
    """Tuned minus default model for each metric and split, plus both models' Train - CV gaps."""
    mean, _ = cv_mean_sd(cv_folds)
    rows = []
    for metric in metrics:
        row = {'metric': metric}
        for split in splits:
            if split == 'CV':
                a, b = mean.loc[tuned, metric], mean.loc[default, metric]
            else:
                lookup = results.set_index(['model', 'split'])[metric]
                a, b = lookup[(tuned, split)], lookup[(default, split)]
            row[f'{split}: {tuned}'], row[f'{split}: {default}'] = a, b
            row[f'{split}: gain'] = a - b
        gaps = overfitting_gaps(results, cv_folds, [metric]).set_index('model')[metric]
        row[f'Train - CV gap: {tuned}'], row[f'Train - CV gap: {default}'] = gaps[tuned], gaps[default]
        rows.append(row)
    return pd.DataFrame(rows)


def drift_table(results, metrics, periods=('June', 'July', 'August')):
    """Long table of each metric by model and monitoring period, with change since June."""
    subset = results[results['split'].isin(periods)]
    rows = []
    for model, frame in subset.groupby('model', sort=False):
        frame = frame.set_index('split')
        for metric in metrics:
            row = {'model': model, 'metric': metric}
            row.update({p: frame.loc[p, metric] for p in periods if p in frame.index})
            row['change_june_to_last'] = row[periods[-1]] - row[periods[0]]
            rows.append(row)
    return pd.DataFrame(rows)


def classification_success_criteria(results, linear_models, forest_model, splits=('June', 'Validation'),
                                    lift_target=2.0, rf_precision_ratio=1.10):
    """Pass/fail table: top-10% lift >= 2 for every model; forest precision >= 1.10 x best linear."""
    lookup = results.set_index(['model', 'split'])
    rows = []
    for split in splits:
        for model in results['model'].unique():
            if model.startswith('C0'):
                continue
            lift = lookup.loc[(model, split), 'top10_lift']
            rows.append({'split': split, 'criterion': f'Top-10% lift >= {lift_target:g}', 'model': model,
                         'value': lift, 'threshold': lift_target, 'passed': bool(lift >= lift_target)})
        best_linear = max(linear_models, key=lambda m: lookup.loc[(m, split), 'precision_top10'])
        threshold = rf_precision_ratio * lookup.loc[(best_linear, split), 'precision_top10']
        for model in forest_model:
            value = lookup.loc[(model, split), 'precision_top10']
            rows.append({'split': split,
                         'criterion': f'Top-10% precision >= {rf_precision_ratio:g} x best linear ({best_linear})',
                         'model': model, 'value': value, 'threshold': threshold,
                         'passed': bool(value >= threshold)})
    return pd.DataFrame(rows)


def score_split(model, split, y_true, prob, order_ids=None):
    """One results row: `classification_eval_metrics` for a model on one split."""
    return {'model': model, 'split': split, **classification_eval_metrics(y_true, prob, order_ids=order_ids)}


def score_months(fitted, final_df, orders, feature_cols, months):
    """Daily inference for each 'YYYY-MM' month, labelled at approval day + lookback.

    Returns {month: DataFrame(order_id, predicted_probability, actual_label, ...)}; the
    daily scores are pooled per month before any metric is computed.
    """
    from . import LOOKBACK
    from .inference import month_bounds
    from .labels import attach_prediction_labels

    scored = {}
    for month in months:
        start, end = month_bounds(month)
        daily = run_inference(final_df, feature_cols, start, end, fitted, verbose=False)
        scored[month] = attach_prediction_labels(daily, orders, lookback=LOOKBACK)
    return scored


def late_probability(fitted, X):
    """Predicted probability of the late class (1) from any fitted classifier."""
    return fitted.predict_proba(X)[:, list(fitted.classes_).index(1)]
