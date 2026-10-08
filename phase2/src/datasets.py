"""One shared definition of Train, Validation, CV folds and full history for each task.

Notebooks and `src.rf_search` call `build_task_data`, so they use identical rows.
It follows the earlier feature-selection notebooks and `src.feature_selection.training_rows`.
"""
from dataclasses import dataclass, field

import pandas as pd

from . import LOOKBACK, PXD
from .features import LEAKAGE_COLS, SELECTED_CAT_COLS, SELECTED_NUM_COLS
from .labels import get_required_dates, labels_as_of, regression_targets_as_of
from .splits import (available_outcome_folds, available_training_rows, chronological_split,
                     day_blocked_time_series_folds)

RUN_DATE = '2018-06-02'
WAITING_PERIOD = LOOKBACK
VALIDATION_DAYS = 30
N_CV_FOLDS = 5
CV_VALIDATION_DAYS = 30
EXCLUDED_STATUSES = ['canceled', 'unavailable']  # Never delivered: no delivery outcome
EXPECTED_TRAIN_ROWS = {'classification': 45972, 'regression': 45371}
SCORING = {'classification': 'average_precision', 'regression': 'neg_root_mean_squared_error'}


@dataclass
class TaskData:
    """Rows and folds for one task. `y` is the late label (0/1) or days from the promise."""
    task: str
    scoring: str
    num_cols: list
    cat_cols: list
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    history_df: pd.DataFrame  # All known-outcome rows before the run date (June refit)
    folds: list
    window: tuple = field(default=())

    @property
    def feature_cols(self):
        return self.num_cols + self.cat_cols

    @property
    def X_train(self):
        return self.train_df[self.feature_cols]

    @property
    def y_train(self):
        return self.train_df['y']

    @property
    def X_validation(self):
        return self.validation_df[self.feature_cols]

    @property
    def y_validation(self):
        return self.validation_df['y']

    @property
    def train_ids(self):
        return set(self.train_df['order_id'])

    def counts(self):
        return {'train': len(self.train_df), 'validation': len(self.validation_df),
                'history': len(self.history_df), 'cv_folds': len(self.folds)}


def build_task_data(task, final_df, check_counts=True):
    """Build Train, Validation, full history and the CV folds for 'classification' or 'regression'."""
    if task not in SCORING:
        raise ValueError("task must be 'classification' or 'regression'")
    num_cols, cat_cols = SELECTED_NUM_COLS.copy(), SELECTED_CAT_COLS.copy()
    assert not set(num_cols + cat_cols) & set(LEAKAGE_COLS), 'Outcome columns used as features'
    start, end, _ = get_required_dates(RUN_DATE, lookback=LOOKBACK, pXd=PXD, verbose=False)
    window = final_df[final_df['order_approved_dt'].between(start, end)]

    if task == 'classification':
        audit = labels_as_of(window.drop_duplicates('order_id'), run_date=RUN_DATE)
        history = audit[audit['label_as_of_run'].notna()].copy()
        history['y'] = history['label_as_of_run'].astype(int)

        def target_as_of(rows, date):
            return labels_as_of(rows, run_date=date)

        audit_col = 'label_as_of_run'
    else:
        window = window[~window['order_status'].isin(EXCLUDED_STATUSES)]
        audit = regression_targets_as_of(window, run_date=RUN_DATE, waiting_period=WAITING_PERIOD)
        history = audit[audit['target_as_of_run'].notna()].copy()
        history['y'] = history['target_as_of_run'].astype(float)
        # The late label for the same orders, used by the dual-framing check
        history['late_label'] = labels_as_of(history, run_date=RUN_DATE)['label_as_of_run']

        def target_as_of(rows, date):
            return regression_targets_as_of(rows, date, WAITING_PERIOD)

        audit_col = 'target_as_of_run'

    train_df, validation_df = chronological_split(
        history, date_col='order_approved_dt', test_days=VALIDATION_DAYS, gap_days=WAITING_PERIOD)
    train_df = available_training_rows(
        train_df, validation_df['order_approved_dt'].min(), 'y', target_as_of, audit_col)
    train_df = train_df.sort_values('order_approved_dt', kind='stable')
    folds = day_blocked_time_series_folds(
        train_df['order_approved_dt'], n_splits=N_CV_FOLDS, gap_days=WAITING_PERIOD,
        validation_days=CV_VALIDATION_DAYS)
    folds = available_outcome_folds(train_df, folds, train_df['y'], target_as_of, audit_col)
    assert len(folds) == N_CV_FOLDS
    if task == 'classification':
        assert all(train_df['y'].iloc[tr].nunique() == 2 for tr, _ in folds)
    if check_counts and len(train_df) != EXPECTED_TRAIN_ROWS[task]:
        raise AssertionError(f'{task} Train has {len(train_df):,} rows, '
                             f'expected {EXPECTED_TRAIN_ROWS[task]:,}')
    return TaskData(task, SCORING[task], num_cols, cat_cols, train_df, validation_df,
                    history, folds, (start, end))
