"""Extra evaluation measures for the companion classification notebook.

Scores are evaluated only where the cohort's actual label is known. Capacity is
allocated among every scored order, including orders whose outcome is unknown.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def daily_capacity(predictions, fraction=0.10):
    """Evaluate a fixed fraction of each day's highest-risk scored orders.

    A cutoff tie is split uniformly in expectation. Unknown outcomes keep their
    share of the daily capacity, but do not enter known-outcome coverage.
    """
    if not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0, 1]')
    required = {'approval_date', 'score', 'actual_label'}
    if not required.issubset(predictions.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(predictions.columns))}')
    if predictions.empty:
        return dict(scored=0, selected=0, known=0, known_selected=0,
                    unknown_selected=0, coverage=np.nan, lift=np.nan)
    if predictions['score'].isna().any() or not predictions['score'].between(0, 1).all():
        raise ValueError('Scores must be probabilities between 0 and 1')

    selected = known_selected = late_selected = random_late = 0.0
    for _, day in predictions.groupby('approval_date'):
        count = max(1, math.ceil(len(day) * fraction))
        cutoff = day['score'].nlargest(count).iloc[-1]
        above = day['score'].gt(cutoff).astype(float)
        at_cutoff = day['score'].eq(cutoff)
        weight = above + at_cutoff.astype(float) * ((count - above.sum()) / at_cutoff.sum())
        known = day['actual_label'].notna()
        late = day['actual_label'].eq(1).fillna(False)
        selected += count
        known_selected += weight[known].sum()
        late_selected += weight[late].sum()
        random_late += count / len(day) * late.sum()

    known = predictions['actual_label'].notna()
    late_total = predictions['actual_label'].eq(1).fillna(False).sum()
    return dict(scored=len(predictions), selected=int(selected), known=int(known.sum()),
                known_selected=known_selected, unknown_selected=selected - known_selected,
                coverage=late_selected / late_total if late_total else np.nan,
                lift=late_selected / random_late if random_late else np.nan)


def extra_metrics(predictions, fraction=0.10):
    """Average precision and Brier score plus daily capacity measures."""
    known = predictions.loc[predictions['actual_label'].notna()]
    y = known['actual_label'].astype(int)
    scores = known['score'].to_numpy(dtype=float)
    return {
        'average_precision': average_precision_score(y, scores) if y.nunique() == 2 else np.nan,
        'brier_score': brier_score_loss(y, scores) if len(y) else np.nan,
        'label_availability': len(known) / len(predictions) if len(predictions) else np.nan,
        **daily_capacity(predictions, fraction),
    }


def paired_monthly_ap(predictions, candidate, reference='Logistic Regression', repeats=499,
                      block_days=7, seed=42):
    """Compare equal-month average precision using paired date-block resampling.

    Both models must score the same known orders in every month. This interval
    describes the observed development months, conditional on fitted models.
    """
    required = {'run_date', 'model', 'order_id', 'approval_date', 'actual_label', 'score'}
    if not required.issubset(predictions.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(predictions.columns))}')
    if repeats < 1 or block_days < 1:
        raise ValueError('repeats and block_days must be positive')
    rows = predictions.loc[predictions['model'].isin([candidate, reference]) &
                           predictions['actual_label'].notna()].copy()
    keys = ['run_date', 'order_id', 'approval_date', 'actual_label']
    if rows.duplicated(keys + ['model']).any():
        raise ValueError('Duplicate model predictions')
    pairs = rows.pivot(index=keys, columns='model', values='score').reset_index()
    if candidate not in pairs or reference not in pairs or pairs[[candidate, reference]].isna().any().any():
        raise ValueError('Models must score identical known orders')

    def ap_difference(frame, weight=None):
        y = frame['actual_label'].astype(int)
        if y.nunique() < 2:
            return np.nan
        return (average_precision_score(y, frame[candidate], sample_weight=weight)
                - average_precision_score(y, frame[reference], sample_weight=weight))

    months = [month.copy() for _, month in pairs.groupby('run_date')]
    observed = np.nanmean([ap_difference(month) for month in months])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        month_differences = []
        for month in months:
            days = pd.date_range(month['approval_date'].min(), month['approval_date'].max())
            starts = rng.integers(0, len(days), size=math.ceil(len(days) / block_days))
            sampled = np.concatenate([(start + np.arange(block_days)) % len(days)
                                      for start in starts])[:len(days)]
            counts = np.bincount(sampled, minlength=len(days))
            day_index = pd.Index(days).get_indexer(pd.to_datetime(month['approval_date']))
            month_differences.append(ap_difference(month, counts[day_index]))
        estimates.append(np.nanmean(month_differences))
    low, high = np.nanquantile(estimates, [0.025, 0.975])
    return {'candidate': candidate, 'reference': reference,
            'mean_ap_difference': observed, 'ci_low': low, 'ci_high': high,
            'months': len(months), 'block_days': block_days}
