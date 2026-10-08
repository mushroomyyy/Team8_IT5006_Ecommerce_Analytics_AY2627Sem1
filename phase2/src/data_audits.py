"""Data, label and transformation audits reported in notebook 01.

Each function takes the notebook's own tables and returns a small DataFrame, so
the notebook stays short and the numbers can be tested. Nothing here fits a model.
"""
import numpy as np
import pandas as pd

from .datasets import RUN_DATE
from .features import LEAKAGE_COLS
from .labels import get_required_dates, labels_as_of, regression_targets_as_of
from .linear_transforms import linear_design, skewness_table
from .feature_selection import numeric_vif

# Skewed amounts, ratios, distances and lags that plan section 4.2 allows log1p for.
LOG1P_CANDIDATES = [
    'payment_value_sum', 'order_frieght_value_sum', 'order_product_weight_g_sum',
    'order_product_volume_cm3_sum', 'approval_lag_hours', 'seller_dist_km_mean',
    'freight_to_price_ratio', 'orders_approved_prev_7d',
]
SKEW_LIMIT = 1.0
OUTCOME_ORDER_COLS = ['order_approved_at', 'order_delivered_customer_date',
                      'order_estimated_delivery_date']
EXCLUDED_STATUSES = ['canceled', 'unavailable']


def raw_table_summary(olist):
    """Rows, columns, duplicate keys and missing cells of each raw Olist table."""
    records = []
    for name, frame in olist.items():
        records.append({'table': name, 'rows': len(frame), 'columns': frame.shape[1],
                        'missing_cells': int(frame.isna().sum().sum()),
                        'duplicate_rows': int(frame.duplicated().sum())})
    return pd.DataFrame(records)


def cleaning_steps(olist, final_df):
    """Row counts from the raw orders table to the order-level feature table."""
    orders = olist['orders']
    approved = orders['order_approved_at'].notna()
    reversed_dates = (orders['order_delivered_customer_date'].dt.normalize()
                      < orders['order_approved_at'].dt.normalize())
    assert final_df['order_id'].is_unique, 'Feature table must have one row per order'
    in_table = final_df['order_id']
    return pd.DataFrame([
        ('Orders in the raw orders table', len(orders)),
        ('Without an approval timestamp (dropped: no prediction point)', int((~approved).sum())),
        ('Approved orders', int(approved.sum())),
        ('Approved before January 2017 (dropped: sparse early months)', int(approved.sum()) - len(final_df)),
        ('Order-level feature table (one row per order)', len(final_df)),
        ('  of which delivered before approval (kept; label rules mark them invalid)',
         int(in_table.isin(orders.loc[reversed_dates, 'order_id']).sum())),
        ('  of which canceled or unavailable (dropped from regression only)',
         int(final_df['order_status'].isin(EXCLUDED_STATUSES).sum())),
    ], columns=['step', 'orders'])


def feature_missingness(final_df, columns):
    """Missing count and share for each modelling feature in the full feature table."""
    missing = final_df[columns].isna().sum()
    table = pd.DataFrame({'missing': missing, 'share_pct': 100 * missing / len(final_df)})
    return table[table['missing'] > 0].sort_values('missing', ascending=False)


def population_funnel(task_data, final_df):
    """Orders at each step from the 365-day window down to Train, for one task."""
    start, end = task_data.window
    window = final_df[final_df['order_approved_dt'].between(start, end)]
    steps = [('Approved orders in the 365-day window', len(window))]
    if task_data.task == 'regression':
        window = window[~window['order_status'].isin(EXCLUDED_STATUSES)]
        steps.append(('After dropping canceled and unavailable orders', len(window)))
    history, train, validation = task_data.history_df, task_data.train_df, task_data.validation_df
    steps += [('Outcome known on the run date (history)', len(history)),
              ('Validation (newest 30 approval days)', len(validation)),
              ('Excluded: 45-day gap and outcomes not yet known', len(history) - len(validation) - len(train)),
              ('Train', len(train))]
    return pd.DataFrame(steps, columns=['step', 'orders']).assign(task=task_data.task)


def target_summary(task_data):
    """Target distribution in Train, Validation and the full history."""
    records = []
    for name, rows in [('Train', task_data.train_df), ('Validation', task_data.validation_df),
                       ('History', task_data.history_df)]:
        y = rows['y']
        if task_data.task == 'classification':
            records.append({'split': name, 'orders': len(y), 'late_orders': int(y.sum()),
                            'late_rate_pct': 100 * y.mean()})
        else:
            records.append({'split': name, 'orders': len(y), 'mean_days': y.mean(),
                            'median_days': y.median(), 'sd_days': y.std(),
                            'late_share_pct': 100 * (y > 0).mean()})
    return pd.DataFrame(records)


def buffer_label_audit(orders, order_ids, run_dates, buffer_days, window_days, target_pct=99.0):
    """Label completeness for each run date and candidate buffer length.

    A buffer is the number of days between the newest training approval and the
    inference date. Returns (review, summary): one review row per run date, buffer
    and scope (full window or newest 7 days), and the worst completeness per buffer.
    """
    label_orders = orders.loc[orders['order_id'].isin(order_ids),
                              ['order_id'] + OUTCOME_ORDER_COLS].drop_duplicates('order_id').copy()
    label_orders['order_approved_dt'] = label_orders['order_approved_at'].dt.normalize()
    records = []
    for run_date in run_dates:
        run_day = pd.Timestamp(run_date).normalize()
        as_of = labels_as_of(label_orders, run_date=run_date)
        for buffer in buffer_days:
            end = run_day - pd.Timedelta(days=1 + buffer)
            start = end - pd.Timedelta(days=window_days - 1)
            window = as_of[as_of['order_approved_dt'].between(start, end)]
            for scope, scope_start in [('full window', start),
                                       ('newest 7 days', max(start, end - pd.Timedelta(days=6)))]:
                cohort = window[window['order_approved_dt'].ge(scope_start)]
                known = int(cohort['label_as_of_run'].notna().sum())
                completeness = 100 * known / len(cohort) if len(cohort) else np.nan
                records.append({
                    'run_date': run_day.date(), 'buffer_days': buffer, 'scope': scope,
                    'orders': len(cohort), 'known_labels': known, 'unknown_labels': len(cohort) - known,
                    'late_rate_pct': 100 * cohort['label_as_of_run'].eq(1).sum() / known if known else np.nan,
                    'completeness_pct': completeness, 'meets_target': bool(completeness >= target_pct)})
    review = pd.DataFrame(records)
    summary = review.groupby('buffer_days', as_index=False).agg(
        worst_completeness_pct=('completeness_pct', 'min'),
        meets_target_everywhere=('meets_target', 'all'))
    return review, summary


def label_availability(task_data, final_df):
    """Known and unknown labels on the run date for the classification window."""
    start, end = task_data.window
    window = final_df[final_df['order_approved_dt'].between(start, end)].drop_duplicates('order_id')
    audit = labels_as_of(window, run_date=RUN_DATE)
    counts = audit['label_status'].value_counts().rename_axis('label_status').to_frame('orders')
    counts['share_pct'] = 100 * counts['orders'] / counts['orders'].sum()
    return counts


def waiting_period_audit(final_df, run_dates, waiting_periods, lookback, window_days):
    """Share of training orders capped, and capped before their promise, per waiting period.

    Mirrors `buffer_label_audit` for the regression target: for each run date, take the
    365-day window ending `lookback` days before inference and count how many orders
    are still waiting at the cap `W`.
    """
    records = []
    for run_date in run_dates:
        start, end, _ = get_required_dates(run_date, lookback=lookback, pXd=window_days, verbose=False)
        window = final_df[final_df['order_approved_dt'].between(start, end)
                          & ~final_df['order_status'].isin(EXCLUDED_STATUSES)]
        for waiting_period in waiting_periods:
            targets = regression_targets_as_of(window, run_date=run_date, waiting_period=waiting_period)
            status = targets['target_status']
            capped = status.isin(['capped_delivered', 'capped_undelivered'])
            early = capped & targets['promised_lead_days'].gt(waiting_period)
            records.append({'run_date': run_date, 'waiting_period': waiting_period,
                            'orders': len(targets),
                            'known_pct': 100 * targets['target_as_of_run'].notna().mean(),
                            'capped_pct': 100 * capped.mean(),
                            'capped_before_promise_pct': 100 * early.mean(),
                            'unresolved_pct': 100 * status.eq('unresolved').mean()})
    return pd.DataFrame(records)


def delivered_lead_summary(final_df):
    """Distribution of lead time and days from the promise for delivered orders."""
    delivered = final_df['order_delivered_customer_date'].notna()
    lead = (final_df['order_delivered_customer_date'].dt.normalize() - final_df['order_approved_dt']).dt.days
    return (pd.DataFrame({'lead_days': lead[delivered],
                          'days_from_promise': (lead - final_df['promised_lead_days'])[delivered]})
            .describe(percentiles=[.05, .25, .5, .75, .95, .99]).T)


def skewness_report(train_rows, num_cols, limit=SKEW_LIMIT, candidates=LOG1P_CANDIDATES):
    """Train skewness before and after log1p for every numeric feature.

    `log1p_applied` is true for plan candidates with |skew| > `limit` (non-negative only).
    """
    table = skewness_table(train_rows, num_cols, limit)
    table['candidate'] = table['feature'].isin(candidates)
    table['log1p_applied'] = table['candidate'] & table['log1p_recommended']
    return table.drop(columns='log1p_recommended')


def route_audit(train_rows, task):
    """Orders and outcome by route type, plus the multi-state-seller group.

    Classification reports late orders; regression reports the mean days from the promise.
    """
    groups = {level: rows for level, rows in train_rows.groupby('route_type')}
    groups['seller_state_count >= 2'] = train_rows[train_rows['seller_state_count'] >= 2]
    records = []
    for name, rows in groups.items():
        record = {'group': name, 'orders': len(rows)}
        if task == 'classification':
            record.update(late_orders=int(rows['y'].sum()), late_rate_pct=100 * rows['y'].mean())
        else:
            record.update(mean_days_from_promise=rows['y'].mean())
        records.append(record)
    return pd.DataFrame(records)


def linear_design_vif(task_data, log_cols, cyclic=True, merge_mixed_route=True):
    """VIF of every encoded column of the linear design fitted on Train."""
    assert not set(task_data.feature_cols) & set(LEAKAGE_COLS), 'Outcome columns used as features'
    design, _, _ = linear_design(task_data.train_df, task_data.num_cols, task_data.cat_cols,
                                 log_cols=log_cols, cyclic=cyclic, merge_mixed_route=merge_mixed_route)
    vif = numeric_vif(design)
    return pd.DataFrame({'task': task_data.task, 'feature': vif.index, 'vif': vif.to_numpy()})
