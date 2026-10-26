"""Export the tables behind the app's Explore the data page, using Phase 2 definitions.

Run from repository root: python -m deployment.export.export_eda
Needs the raw Olist tables (Olist_CSV/ or the bundled zip). The app then reads
only the small files written here, never the raw data.

Definitions follow the Phase 2 report: approved orders (approval Jan 2017 to
Aug 2018), approval date as the time axis, late labels evaluated at approval +
45 days (late if delivered after the promised day or still undelivered once it
has passed; unknown otherwise), and days from promise = min(lead days, 45) -
promised lead days. Canceled/unavailable orders have no delivery outcome, so they are
neither late nor on time and carry no late label or days from promise.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_olist_data
from src.features import add_extra_features, build_feature_table
from src.labels import attach_prediction_labels, attach_regression_actuals

OUT = ROOT / 'deployment/app/data/eda'
FIRST_MONTH, LAST_MONTH = '2017-01', '2018-08'


def order_table(data, frame):
    """One row per approved order with Phase 2 features, labels and descriptive columns."""
    orders = frame.loc[frame['partition_yyyymm'].between(FIRST_MONTH, LAST_MONTH)].copy()
    labels = attach_prediction_labels(orders[['order_id']].assign(predicted_probability=0.0), data['orders'], lookback=45)
    actuals = attach_regression_actuals(orders[['order_id']], data['orders'], waiting_period=45)
    canceled = orders['order_status'].isin(['canceled', 'unavailable'])

    items = (data['order_items'].merge(data['products'][['product_id', 'product_category_name']], on='product_id', how='left')
             .merge(data['sellers'][['seller_id', 'seller_state']], on='seller_id', how='left')
             .sort_values(['order_id', 'order_item_id']))
    first_item = items.groupby('order_id').agg(primary_category=('product_category_name', 'first'),
                                               primary_seller_state=('seller_state', 'first'))
    reviews = data['reviews'].groupby('order_id').agg(review_score=('review_score', 'mean'),
                                                      has_review_comment=('review_comment_message', lambda v: v.notna().any()))

    out = pd.DataFrame({
        'order_id': orders['order_id'].values,
        'approved_date': orders['order_approved_dt'].values,
        'purchase_weekday': orders['order_purchase_timestamp'].dt.dayofweek.add(1).values,
        'purchase_hour': orders['order_purchase_timestamp'].dt.hour.values,
        'order_status': orders['order_status'].values,
        'customer_state': orders['customer_state'].values,
        'route_type': orders['route_type'].values,
        'seller_dist_km_mean': orders['seller_dist_km_mean'].values,
        'item_count': orders['order_item_count'].values,
        'seller_count': orders['order_seller_count'].values,
        'category_count': orders['order_product_category_count'].values,
        'order_value': orders['order_pdt_price_sum'].values,
        'freight_value': orders['order_frieght_value_sum'].values,
        'payment_value': orders['payment_value_sum'].values,
        'payment_combination': orders['payment_combination'].values,
        'payment_installments_max': orders['payment_installments_max'].values,
        'promised_lead_days': orders['promised_lead_days'].values,
        'approval_lag_hours': orders['approval_lag_hours'].values,
        'approval_to_carrier_days': orders['durn_to_carrier_days'].values,
        'approval_to_customer_days': orders['durn_to_customer_days'].values,
    })
    out['approved_month'] = pd.to_datetime(out['approved_date']).dt.to_period('M').dt.to_timestamp()
    out = out.join(first_item, on='order_id').join(reviews, on='order_id')
    label_map = labels.set_index('order_id')
    out['late'] = out['order_id'].map(label_map['actual_label']).astype('Int8')
    out['label_status'] = out['order_id'].map(label_map['label_status'])
    out.loc[canceled.values, 'late'] = pd.NA
    out.loc[canceled.values, 'label_status'] = 'canceled_or_unavailable'
    target = actuals.set_index('order_id')['actual_target'].astype('Float64')
    out['days_from_promise'] = out['order_id'].map(target)
    out.loc[canceled.values, 'days_from_promise'] = pd.NA
    out['review_group'] = pd.cut(out['review_score'], bins=[0, 2, 4, 5], labels=['Low (1-2)', 'Mid (3-4)', 'High (5)']).astype('string')
    for column in ['order_status', 'customer_state', 'route_type', 'payment_combination', 'primary_category',
                   'primary_seller_state', 'label_status']:
        out[column] = out[column].astype('category')
    return out


def seller_history(data, approved):
    """Leakage-safe seller track record: only deliveries completed before each order's approval."""
    orders = data['orders'].loc[data['orders']['order_id'].isin(approved['order_id'])]
    pairs = data['order_items'][['order_id', 'seller_id']].drop_duplicates().merge(
        orders[['order_id', 'order_status', 'order_approved_at', 'order_delivered_customer_date',
                'order_estimated_delivery_date']], on='order_id')
    delivered = pairs['order_status'].eq('delivered') & pairs['order_delivered_customer_date'].notna() & pairs['order_estimated_delivery_date'].notna()
    pairs = pairs.loc[delivered].copy()
    # Phase 2 label for delivered orders: delivered after the promised calendar day.
    pairs['late'] = (pairs['order_delivered_customer_date'].dt.normalize() > pairs['order_estimated_delivery_date'].dt.normalize()).astype(int)
    events = (pairs.groupby(['seller_id', 'order_delivered_customer_date'], as_index=False)
              .agg(completed=('order_id', 'nunique'), late_count=('late', 'sum'))
              .sort_values(['seller_id', 'order_delivered_customer_date']))
    events['prior_completed_orders'] = events.groupby('seller_id')['completed'].cumsum()
    events['prior_late_rate'] = events.groupby('seller_id')['late_count'].cumsum() / events['prior_completed_orders']
    history = pd.merge_asof(
        pairs[['order_id', 'seller_id', 'order_approved_at', 'late']].sort_values('order_approved_at'),
        events[['seller_id', 'order_delivered_customer_date', 'prior_completed_orders', 'prior_late_rate']]
        .rename(columns={'order_delivered_customer_date': 'available_at'}).sort_values('available_at'),
        by='seller_id', left_on='order_approved_at', right_on='available_at', direction='backward', allow_exact_matches=False)
    history['prior_completed_orders'] = history['prior_completed_orders'].fillna(0).astype(int)
    return history[['order_id', 'seller_id', 'late', 'prior_completed_orders', 'prior_late_rate']]


def repeat_monthly(data, approved):
    """Share of each month's customers who place another approved order within three months."""
    customers = data['orders'][['order_id', 'customer_id']].merge(data['customers'][['customer_id', 'customer_unique_id']], on='customer_id')
    months = approved[['order_id', 'approved_month']].merge(customers, on='order_id')[['customer_unique_id', 'approved_month']].drop_duplicates()
    later = pd.concat([months.assign(approved_month=months['approved_month'] - pd.DateOffset(months=k)) for k in (1, 2, 3)]).drop_duplicates()
    later['repeat'] = 1
    months = months.merge(later, on=['customer_unique_id', 'approved_month'], how='left').fillna({'repeat': 0})
    table = months.groupby('approved_month', as_index=False).agg(customers=('customer_unique_id', 'nunique'), repeat_customers=('repeat', 'sum'))
    table['repeat_rate'] = table['repeat_customers'] / table['customers']
    # The last three months cannot show a full three-month follow-up window.
    table['complete_window'] = table['approved_month'] <= pd.Timestamp(LAST_MONTH) - pd.DateOffset(months=3)
    return table


def data_checks(data):
    """Raw-table inventory, join checks and monthly coverage, as small JSON-ready tables."""
    inventory = [{'table': name, 'rows': int(len(df)), 'columns': int(df.shape[1]),
                  'missing_cells_pct': round(float(100 * df.isna().sum().sum() / df.size), 2)}
                 for name, df in data.items() if name != 'category_translation']
    items, orders = data['order_items'], data['orders']
    checks = [
        {'check': 'Order items without a matching order', 'result': int((~items['order_id'].isin(orders['order_id'])).sum())},
        {'check': 'Orders without any line items', 'result': int((~orders['order_id'].isin(items['order_id'])).sum())},
        {'check': 'Orders without approval (excluded)', 'result': int(orders['order_approved_at'].isna().sum())},
        {'check': 'Median / maximum items per order', 'result': f"{items.groupby('order_id').size().median():.0f} / {items.groupby('order_id').size().max()}"},
        {'check': 'Median / maximum payments per order', 'result': f"{data['payments'].groupby('order_id').size().median():.0f} / {data['payments'].groupby('order_id').size().max()}"},
    ]
    month = orders['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
    coverage = (pd.DataFrame({'month': month, 'approved': orders['order_approved_at'].notna(), 'delivered': orders['order_delivered_customer_date'].notna()})
                .groupby('month').agg(orders=('approved', 'size'), approved=('approved', 'sum'), delivered=('delivered', 'sum')).reset_index())
    coverage['month'] = coverage['month'].dt.strftime('%Y-%m')
    return {'inventory': inventory, 'checks': checks, 'coverage': coverage.to_dict(orient='records'),
            'window': {'first_month': FIRST_MONTH, 'last_month': LAST_MONTH}}


def export():
    data = load_olist_data(verbose=False)
    frame = add_extra_features(build_feature_table(data), data)
    approved = order_table(data, frame)
    OUT.mkdir(parents=True, exist_ok=True)
    approved.to_parquet(OUT / 'orders.parquet', index=False, compression='zstd')
    seller_history(data, approved).to_parquet(OUT / 'seller_history.parquet', index=False, compression='zstd')
    repeat_monthly(data, approved).to_parquet(OUT / 'repeat_monthly.parquet', index=False, compression='zstd')
    (OUT / 'data_checks.json').write_text(json.dumps(data_checks(data), indent=1) + '\n')
    manifest_path = ROOT / 'deployment/artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['eda_data'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    manifest['eda_export'] = 'python -m deployment.export.export_eda'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    known = approved['late'].notna()
    print(f"{len(approved):,} approved orders {FIRST_MONTH}..{LAST_MONTH}; late rate {approved.loc[known, 'late'].mean():.2%} of {known.sum():,} known")
    for p in sorted(OUT.iterdir()): print(f'  {p.name}: {p.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    export()
