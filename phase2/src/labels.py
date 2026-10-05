"""Run-date windows and as-of-run-date targets.

`get_required_dates`, `labels_as_of` and `attach_prediction_labels` come from
model_dev_v2.ipynb. `regression_targets_as_of` and `attach_regression_actuals`
apply the same date rules to the regression target: days early or late against the
estimated delivery date.
"""
import pandas as pd

OUTCOME_COLS = ['order_id', 'order_approved_at', 'order_delivered_customer_date',
                'order_estimated_delivery_date']


def _parse_run_date(run_date):
    if not isinstance(run_date, str):
        raise ValueError('run_date must be a YYYY-MM-DD string')
    run_day = pd.to_datetime(run_date, format='%Y-%m-%d', errors='coerce')
    if pd.isna(run_day) or run_day.strftime('%Y-%m-%d') != run_date:
        raise ValueError('run_date must be a valid date in YYYY-MM-DD format')
    return run_day


def _check_non_negative_int(value, name, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"non-negative" if allow_zero else "positive"} integer')


def get_required_dates(run_date: str, lookback: int = 45, pXd: int = 180, verbose=True):
    """Return (start, end, inference_date) as YYYY-MM-DD strings; window is inclusive."""
    _check_non_negative_int(pXd, 'pXd', allow_zero=False)
    _check_non_negative_int(lookback, 'lookback')
    run_day = _parse_run_date(run_date)
    inference_date = run_day - pd.Timedelta(days=1)
    end_date = inference_date - pd.Timedelta(days=lookback)
    start_date = end_date - pd.Timedelta(days=pXd - 1)
    if verbose:
        print(f'Train-test period for the inference date of {inference_date:%Y-%m-%d} '
              f'with buffer of {lookback} days:')
        print(f'    {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} ({pXd} days)')
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), inference_date.strftime('%Y-%m-%d')


def labels_as_of(order_rows, run_date: str):
    """Assign late-delivery labels using calendar dates strictly before run_date."""
    run_day = _parse_run_date(run_date)
    result = order_rows.copy()
    approved = pd.to_datetime(result['order_approved_at']).dt.normalize()
    delivered = pd.to_datetime(result['order_delivered_customer_date']).dt.normalize()
    deadline = pd.to_datetime(result['order_estimated_delivery_date']).dt.normalize()
    valid = approved.notna() & approved.lt(run_day) & deadline.notna()
    valid &= ~(delivered.notna() & delivered.lt(approved))  # Delivery before approval
    delivered_by_run = delivered.notna() & delivered.lt(run_day)
    overdue = deadline.lt(run_day)

    result['delivered_by_run'] = delivered_by_run
    result['label_as_of_run'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    result['label_status'] = 'unresolved'
    result.loc[~valid, 'label_status'] = 'invalid_or_missing_dates'
    on_time = valid & delivered_by_run & delivered.le(deadline)
    late_delivered = valid & delivered_by_run & delivered.gt(deadline)
    late_undelivered = valid & ~delivered_by_run & overdue
    result.loc[on_time, 'label_as_of_run'] = 0
    result.loc[late_delivered | late_undelivered, 'label_as_of_run'] = 1
    result.loc[on_time, 'label_status'] = 'on_time_delivered'
    result.loc[late_delivered, 'label_status'] = 'late_delivered'
    result.loc[late_undelivered, 'label_status'] = 'late_undelivered'
    return result


def regression_targets_as_of(order_rows, run_date: str, horizon: int):
    """Assign the days from the estimated delivery date; negative is early, positive is late.

    target = min(delivered - approved, horizon) - (estimated - approved), in calendar days.
    For an order delivered within `horizon` days this is simply delivered - estimated.

    Uses only dates strictly before run_date. An undelivered order is still known to
    have taken at least `run_date - approved` days, so its capped lead time is known
    (= horizon) once that span reaches the horizon. This keeps slow orders in training
    instead of silently dropping them, which would bias the target downward.

    target_status:
      delivered_within_horizon  delivered by the run date within `horizon` days
      capped_delivered          delivered by the run date after more than `horizon` days
      capped_undelivered        not delivered by the run date, already `horizon` days old
      unresolved                not delivered by the run date and younger than `horizon`
      invalid_or_missing_dates  missing approval or estimate, or delivery before approval
    """
    _check_non_negative_int(horizon, 'horizon', allow_zero=False)
    run_day = _parse_run_date(run_date)
    result = order_rows.copy()
    approved = pd.to_datetime(result['order_approved_at']).dt.normalize()
    delivered = pd.to_datetime(result['order_delivered_customer_date']).dt.normalize()
    deadline = pd.to_datetime(result['order_estimated_delivery_date']).dt.normalize()
    valid = approved.notna() & approved.lt(run_day) & deadline.notna()
    valid &= ~(delivered.notna() & delivered.lt(approved))
    delivered_by_run = delivered.notna() & delivered.lt(run_day)
    lead_days = (delivered - approved).dt.days
    promised_days = (deadline - approved).dt.days
    # Days already observed without delivery: approval day through run_date - 1
    elapsed_days = (run_day - approved).dt.days

    within = valid & delivered_by_run & lead_days.le(horizon)
    capped_delivered = valid & delivered_by_run & lead_days.gt(horizon)
    capped_undelivered = valid & ~delivered_by_run & elapsed_days.ge(horizon)

    result['lead_days_as_of_run'] = lead_days.where(valid & delivered_by_run).astype('Float64')
    result['capped_lead_as_of_run'] = pd.Series(pd.NA, index=result.index, dtype='Float64')
    result.loc[within, 'capped_lead_as_of_run'] = lead_days[within].astype(float)
    result.loc[capped_delivered | capped_undelivered, 'capped_lead_as_of_run'] = float(horizon)
    result['target_as_of_run'] = result['capped_lead_as_of_run'] - promised_days
    result['target_status'] = 'unresolved'
    result.loc[~valid, 'target_status'] = 'invalid_or_missing_dates'
    result.loc[within, 'target_status'] = 'delivered_within_horizon'
    result.loc[capped_delivered, 'target_status'] = 'capped_delivered'
    result.loc[capped_undelivered, 'target_status'] = 'capped_undelivered'
    return result


def _outcomes_for(orders, order_ids):
    outcomes = orders.loc[orders['order_id'].isin(order_ids), OUTCOME_COLS]
    outcomes = outcomes.drop_duplicates('order_id').copy()
    for column in OUTCOME_COLS[1:]:
        outcomes[column] = pd.to_datetime(outcomes[column])
    return outcomes


def attach_prediction_labels(predictions, orders, lookback=45, prediction_threshold=0.5):
    """Return a labelled copy of classification predictions.

    Each daily cohort is evaluated at its own inclusive cutoff: approval day + lookback.
    Requires order_id and predicted_probability columns.
    """
    _check_non_negative_int(lookback, 'lookback')
    labelled = predictions.copy()
    outcomes = _outcomes_for(orders, labelled['order_id'])
    outcomes['evaluation_cutoff'] = (outcomes['order_approved_at'].dt.normalize()
                                     + pd.Timedelta(days=lookback))
    outcomes['actual_label'] = pd.Series(pd.NA, index=outcomes.index, dtype='Int64')
    outcomes['label_status'] = 'invalid_or_missing_dates'
    for cutoff, cohort in outcomes.groupby('evaluation_cutoff'):
        # labels_as_of excludes its run date; use the next day to include the cutoff day
        audit = labels_as_of(cohort, run_date=(cutoff + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        outcomes.loc[audit.index, 'actual_label'] = audit['label_as_of_run']
        outcomes.loc[audit.index, 'label_status'] = audit['label_status']

    lookup = outcomes.set_index('order_id')
    for column in ['actual_label', 'label_status', 'evaluation_cutoff']:
        labelled[column] = labelled['order_id'].map(lookup[column])
    labelled['actual_label'] = labelled['actual_label'].astype('Int64')
    labelled['predicted_label'] = (labelled['predicted_probability'] >= prediction_threshold).astype(int)
    return labelled


def attach_regression_actuals(predictions, orders, horizon):
    """Return a copy of regression predictions with their actual days from the estimate.

    Each daily cohort is evaluated at approval day + horizon (inclusive), when every
    valid order's target (with its lead time capped at `horizon`) is known.
    """
    _check_non_negative_int(horizon, 'horizon', allow_zero=False)
    labelled = predictions.copy()
    outcomes = _outcomes_for(orders, labelled['order_id'])
    outcomes['evaluation_cutoff'] = (outcomes['order_approved_at'].dt.normalize()
                                     + pd.Timedelta(days=horizon))
    outcomes['actual_target'] = pd.Series(pd.NA, index=outcomes.index, dtype='Float64')
    outcomes['target_status'] = 'invalid_or_missing_dates'
    for cutoff, cohort in outcomes.groupby('evaluation_cutoff'):
        audit = regression_targets_as_of(
            cohort, run_date=(cutoff + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), horizon=horizon)
        outcomes.loc[audit.index, 'actual_target'] = audit['target_as_of_run']
        outcomes.loc[audit.index, 'target_status'] = audit['target_status']

    lookup = outcomes.set_index('order_id')
    for column in ['actual_target', 'target_status', 'evaluation_cutoff']:
        labelled[column] = labelled['order_id'].map(lookup[column])
    labelled['actual_target'] = labelled['actual_target'].astype('Float64')
    return labelled
