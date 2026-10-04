"""Time-aware train/test splits and CV folds.

A random stratified split inside the train-test window puts later orders into
training and earlier ones into test, so test scores are optimistic. Here the test
set is always the newest approval days, and each CV fold validates on days after
the days it trains on.
"""
import numpy as np
import pandas as pd


def chronological_split(df, date_col='order_approved_dt', test_days=30):
    """Split `df` into (train, test): test holds the last `test_days` approval days."""
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    test_start = dates.max() - pd.Timedelta(days=test_days - 1)
    train, test = df.loc[dates.lt(test_start)].copy(), df.loc[dates.ge(test_start)].copy()
    assert pd.to_datetime(train[date_col]).max() < pd.to_datetime(test[date_col]).min()
    return train, test


def day_blocked_time_series_folds(dates, n_splits=5, gap_days=0, min_train_days=None):
    """Expanding-window CV folds that never split one approval day across train and validation.

    `dates` must be positionally aligned with the X/y passed to the estimator.
    Returns a list of (train_idx, valid_idx) arrays usable as `cv=` in scikit-learn.
    `gap_days` leaves calendar days out between train and validation.
    `min_train_days` reserves an initial training period before that gap, useful
    for short windows. Empty folds raise an error rather than silently disappearing.
    """
    if n_splits < 2 or gap_days < 0:
        raise ValueError('Require at least two folds and a non-negative gap')
    days = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    if np.isnat(days).any():
        raise ValueError('Approval dates must be known')
    unique_days = np.sort(np.unique(days))
    if min_train_days is None:
        blocks = np.array_split(unique_days, n_splits + 1)[1:]
    else:
        if min_train_days < 1:
            raise ValueError('min_train_days must be positive')
        valid_start = unique_days[0] + np.timedelta64(min_train_days + gap_days, 'D')
        blocks = np.array_split(unique_days[unique_days >= valid_start], n_splits)
    folds = []
    for valid_days in blocks:
        if not len(valid_days):
            raise ValueError('Too few validation days for the requested folds')
        train_end = valid_days[0] - np.timedelta64(gap_days, 'D')
        train_idx = np.flatnonzero(days < train_end)
        valid_idx = np.flatnonzero(np.isin(days, valid_days))
        if not len(train_idx) or not len(valid_idx):
            raise ValueError('Empty CV fold: increase the initial training period or reduce folds')
        folds.append((train_idx, valid_idx))
    return folds


def available_outcome_folds(rows, folds, targets, target_as_of, target_col):
    """Retain training rows whose target was already known at validation start.

    The callback uses the existing label rules. Arrays remain positional relative
    to the original X/y; validation outcomes are used only for scoring.
    """
    expected = pd.Series(targets).reset_index(drop=True)
    if len(expected) != len(rows):
        raise ValueError('Targets must align positionally with rows')
    result = []
    for train_idx, valid_idx in folds:
        valid_start = pd.to_datetime(rows['order_approved_dt'].iloc[valid_idx]).min().normalize()
        audit = target_as_of(rows.iloc[train_idx], valid_start.strftime('%Y-%m-%d'))
        known = audit[target_col].reset_index(drop=True)
        available = known.notna() & known.eq(expected.iloc[train_idx].reset_index(drop=True))
        train_idx = train_idx[available.fillna(False).to_numpy(dtype=bool)]
        if not len(train_idx):
            raise ValueError('No training outcomes available at validation start')
        result.append((train_idx, valid_idx))
    return result
