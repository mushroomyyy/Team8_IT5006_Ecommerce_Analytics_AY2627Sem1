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


def day_blocked_time_series_folds(dates, n_splits=5, gap_days=0):
    """Expanding-window CV folds that never split one approval day across train and validation.

    `dates` must be positionally aligned with the X/y passed to the estimator.
    Returns a list of (train_idx, valid_idx) arrays usable as `cv=` in scikit-learn.
    `gap_days` leaves days out between train and validation, e.g. to mimic the lookback.
    """
    days = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    unique_days = np.sort(np.unique(days))
    blocks = np.array_split(unique_days, n_splits + 1)
    folds = []
    for k in range(1, n_splits + 1):
        valid_days = blocks[k]
        train_end = valid_days[0] - np.timedelta64(gap_days, 'D')
        train_idx = np.flatnonzero(days < train_end)
        valid_idx = np.flatnonzero(np.isin(days, valid_days))
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds
