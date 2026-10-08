"""Daily classification inference over a date range, from the earlier notebooks' run_inference."""
from calendar import monthrange

import pandas as pd


def month_bounds(month):
    """First and last day of a 'YYYY-MM' month as YYYY-MM-DD strings."""
    start = pd.Timestamp(f'{month}-01')
    return start.strftime('%Y-%m-%d'), start.replace(day=monthrange(start.year, start.month)[1]).strftime('%Y-%m-%d')


def run_inference(final_df, feature_cols, start_date, end_date, model, verbose=True):
    """Score eligible orders day by day with a fitted classifier (no refitting).

    An order is scored on its approval day if it is not yet delivered by that day.
    Returns order_approved_dt, order_id and the late-class probability for each order;
    pool the days before computing metrics.
    """
    start_day, end_day = pd.to_datetime(start_date).normalize(), pd.to_datetime(end_date).normalize()
    if pd.isna(start_day) or pd.isna(end_day) or start_day > end_day:
        raise ValueError('Inference dates must be valid and start must not exceed end.')
    late_index = list(model.classes_).index(1)
    approval_days = pd.to_datetime(final_df['order_approved_dt']).dt.normalize()
    delivery_days = pd.to_datetime(final_df['order_delivered_customer_date']).dt.normalize()
    frames = []
    for day in pd.date_range(start_day, end_day, freq='D'):
        eligible = approval_days.eq(day) & (delivery_days.isna() | delivery_days.gt(day))
        orders = final_df.loc[eligible]
        if orders.empty:
            continue
        daily = orders[['order_approved_dt', 'order_id']].copy()
        daily['order_approved_dt'] = day
        daily['predicted_probability'] = model.predict_proba(orders[feature_cols])[:, late_index]
        frames.append(daily)
    if frames:
        result = pd.concat(frames, ignore_index=True).sort_values(
            ['order_approved_dt', 'order_id'], ignore_index=True)
    else:
        result = pd.DataFrame({'order_approved_dt': pd.Series(dtype='datetime64[ns]'),
                               'order_id': pd.Series(dtype='object'),
                               'predicted_probability': pd.Series(dtype='float64')})
    if verbose:
        print(f'{len(result):,} orders scored between {start_day:%Y-%m-%d} and {end_day:%Y-%m-%d}')
    return result
