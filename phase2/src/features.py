"""Order-level feature table (model_dev_v2.ipynb cells 14-26) plus extra leakage-safe features.

`build_feature_table` reproduces v2's `final_df` exactly. `add_extra_features` appends
new columns that are all known when the order is approved, and never reads
`order_delivered_carrier_date` or `order_delivered_customer_date`.
"""
import numpy as np
import pandas as pd

# Base feature set used by model_dev_v2.ipynb
NUM_COLS = [
    'order_approved_day_of_week',
    'order_approved_day_of_month',
    'order_approved_month',
    'order_approved_week_of_year',
    'order_item_count',
    'order_seller_count',
    'order_pdt_price_sum',
    'order_frieght_value_sum',
    'order_revenue_sum',
    'order_product_weight_g_sum',
    'order_product_category_count',
    'seller_dist_km_max',
    'seller_dist_km_min',
    'seller_dist_km_median',
    'seller_dist_km_mean',
    'seller_zip_code_prefix_count',
    'seller_city_count',
    'seller_state_count',
    'payment_value_sum',
    'payment_type_value_boleto',
    'payment_type_value_credit_card',
    'payment_type_value_debit_card',
    'payment_type_value_not_defined',
    'payment_type_value_voucher',
    'payment_type_count_boleto',
    'payment_type_count_credit_card',
    'payment_type_count_debit_card',
    'payment_type_count_not_defined',
    'payment_type_count_voucher',
]
CAT_COLS = ['customer_state', 'route_type']

# Added by add_extra_features; each is known at approval time
EXTRA_NUM_COLS = [
    'promised_lead_days',            # Estimated delivery date is shown to the customer at checkout
    'approval_lag_hours',            # Purchase -> approval, both happen before the prediction point
    'order_product_volume_cm3_sum',  # Catalogue dimensions
    'order_product_photos_mean',     # Catalogue attribute
    'freight_to_price_ratio',        # Prices are fixed at checkout
    'payment_installments_max',      # Chosen at checkout
    'is_weekend_approval',
    'is_black_friday_period',        # 2017-11-20 to 2017-11-30 order spike (Phase 1 EDA)
    'is_december_peak',
    'orders_approved_prev_7d',       # Platform load over the 7 completed days before approval
]

# Columns that define the outcome and must never be used as predictors
LEAKAGE_COLS = ['order_delivered_carrier_date', 'order_delivered_customer_date',
                'durn_to_carrier_days', 'durn_to_customer_days', 'order_status']

BLACK_FRIDAY_PERIOD = ('2017-11-20', '2017-11-30')


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in kilometres."""
    earth_radius_km = 6371.0088
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (np.sin(delta_lat / 2) ** 2
         + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def _approved_orders(orders, customers):
    """Orders with an approval date, calendar features and customer location (cell 14)."""
    all_orders = orders[orders['order_approved_at'].notnull()].copy()
    all_orders['order_approved_dt'] = all_orders['order_approved_at'].dt.normalize()
    all_orders['order_approved_day_of_week'] = (all_orders['order_approved_dt'].dt.dayofweek + 1).astype('Int64')
    all_orders['order_approved_day_of_month'] = all_orders['order_approved_dt'].dt.day.astype('Int64')
    all_orders['order_approved_month'] = all_orders['order_approved_dt'].dt.month.astype('Int64')
    all_orders['order_approved_week_of_year'] = (all_orders['order_approved_dt'].dt.isocalendar().week
                                                 .clip(upper=52).astype('Int64'))
    all_orders['partition_yyyymm'] = all_orders['order_approved_at'].dt.to_period('M').astype(str)
    # Outcome durations for analysis only; listed in LEAKAGE_COLS
    all_orders['durn_to_carrier_days'] = (all_orders['order_delivered_carrier_date']
                                          - all_orders['order_approved_at']).dt.total_seconds() / 86400
    all_orders['durn_to_customer_days'] = (all_orders['order_delivered_customer_date']
                                           - all_orders['order_approved_at']).dt.total_seconds() / 86400
    return all_orders.merge(customers, on='customer_id', how='left')


def _route_types(all_orders, order_items, sellers):
    """Tag each order as all-interstate, all-same-state or mixed (cell 16)."""
    items = order_items.merge(sellers[['seller_id', 'seller_state', 'seller_city']],
                              on='seller_id', how='left')
    routes = all_orders.merge(items, on='order_id', how='left')
    routes['item_is_interstate'] = routes['customer_state'] != routes['seller_state']
    states = routes.groupby(['order_id', 'customer_state']).agg(
        all_interstate=('item_is_interstate', 'all'),
        any_interstate=('item_is_interstate', 'any'),
    ).reset_index()
    states['route_type'] = np.select([states['all_interstate'], ~states['any_interstate']],
                                     ['1. All interstate', '2. All same-state'],
                                     default='3. Mixed')
    return states[['order_id', 'route_type']]


def _seller_distances(all_orders, orders, order_items, customers, sellers, geolocation):
    """Seller-to-customer distance aggregated per order (cell 18)."""
    geo_by_zip = geolocation.groupby('geolocation_zip_code_prefix', as_index=False).agg(
        latitude=('geolocation_lat', 'median'),
        longitude=('geolocation_lng', 'median'),
    )
    customer_coordinates = customers[['customer_id', 'customer_zip_code_prefix']].merge(
        geo_by_zip.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
                                   'latitude': 'customer_latitude',
                                   'longitude': 'customer_longitude'}),
        on='customer_zip_code_prefix', how='left')
    seller_coordinates = sellers[['seller_id', 'seller_zip_code_prefix']].merge(
        geo_by_zip.rename(columns={'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
                                   'latitude': 'seller_latitude',
                                   'longitude': 'seller_longitude'}),
        on='seller_zip_code_prefix', how='left')

    # One row per order-seller
    pairs = (order_items[['order_id', 'seller_id']].drop_duplicates()
             .merge(seller_coordinates, on='seller_id', how='left')
             .merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
             .merge(customer_coordinates, on='customer_id', how='left'))
    pairs['distance_km'] = haversine_km(pairs['seller_latitude'], pairs['seller_longitude'],
                                        pairs['customer_latitude'], pairs['customer_longitude'])
    by_order = pairs.groupby('order_id', as_index=False).agg(
        seller_dist_km_max=('distance_km', 'max'),
        seller_dist_km_min=('distance_km', 'min'),
        seller_dist_km_median=('distance_km', 'median'),
        seller_dist_km_mean=('distance_km', 'mean'),
        seller_dist_km_count=('seller_id', 'nunique'),
    )
    return all_orders[['order_id']].merge(by_order, on='order_id', how='left')


def _order_item_aggregates(order_items, products, sellers):
    """Item, product and seller aggregates per order (cells 20-21)."""
    items = order_items.merge(products[['product_id', 'product_category_name',
                                        'product_photos_qty', 'product_weight_g']],
                              on='product_id', how='left')
    items = items.merge(sellers, on='seller_id', how='left')
    items['item_revenue'] = items['price'] + items['freight_value']
    items = items.sort_values(['order_id', 'order_item_id'])
    return items.groupby('order_id').agg(
        order_item_count=('product_id', 'count'),
        order_seller_count=('seller_id', 'nunique'),
        order_pdt_price_sum=('price', 'sum'),
        order_frieght_value_sum=('freight_value', 'sum'),
        order_revenue_sum=('item_revenue', 'sum'),
        order_product_weight_g_sum=('product_weight_g', 'sum'),
        order_product_category_count=('product_category_name', 'nunique'),
        seller_zip_code_prefix_count=('seller_zip_code_prefix', 'nunique'),
        seller_city_count=('seller_city', 'nunique'),
        seller_state_count=('seller_state', 'nunique'),
        product_category_name=('product_category_name', set),
        seller_zip_code_prefix=('seller_zip_code_prefix', set),
        seller_city=('seller_city', set),
        seller_state=('seller_state', set),
    ).reset_index()


def _payment_aggregates(payments):
    """Total payment value plus value and count per payment type (cell 23)."""
    by_type = payments.pivot_table(index='order_id', columns='payment_type',
                                   values='payment_value', aggfunc=['sum', 'size'], fill_value=0)
    by_type.columns = [f"payment_type_{'count' if agg == 'size' else 'value'}_{ptype}"
                       for agg, ptype in by_type.columns]
    return (payments.groupby('order_id')
            .agg(payment_value_sum=('payment_value', 'sum'))
            .join(by_type)
            .reset_index())


def build_feature_table(olist, min_partition='2017-01'):
    """Return v2's `final_df`: one row per approved order from `min_partition` onwards."""
    orders, order_items = olist['orders'], olist['order_items']
    customers, sellers = olist['customers'], olist['sellers']

    all_orders = _approved_orders(orders, customers)
    final_df = (all_orders
                .merge(_route_types(all_orders, order_items, sellers), on='order_id', how='left')
                .merge(_seller_distances(all_orders, orders, order_items, customers, sellers,
                                         olist['geolocation']), on='order_id', how='left')
                .merge(_order_item_aggregates(order_items, olist['products'], sellers),
                       on='order_id', how='left')
                .merge(_payment_aggregates(olist['payments']), on='order_id', how='left'))
    return final_df[final_df['partition_yyyymm'] >= min_partition].reset_index(drop=True)


def add_extra_features(final_df, olist):
    """Return a copy of `final_df` with EXTRA_NUM_COLS appended."""
    df = final_df.copy()
    approved_dt = df['order_approved_dt']

    df['promised_lead_days'] = (df['order_estimated_delivery_date'].dt.normalize()
                                - approved_dt).dt.days
    df['approval_lag_hours'] = (df['order_approved_at']
                                - df['order_purchase_timestamp']).dt.total_seconds() / 3600

    products = olist['products'].copy()
    products['product_volume_cm3'] = (products['product_length_cm']
                                      * products['product_height_cm']
                                      * products['product_width_cm'])
    items = olist['order_items'].merge(
        products[['product_id', 'product_volume_cm3', 'product_photos_qty']],
        on='product_id', how='left')
    item_agg = items.groupby('order_id').agg(
        order_product_volume_cm3_sum=('product_volume_cm3', lambda s: s.sum(min_count=1)),
        order_product_photos_mean=('product_photos_qty', 'mean'),
    )
    installments = olist['payments'].groupby('order_id')['payment_installments'].max()
    df = df.join(item_agg, on='order_id')
    df['payment_installments_max'] = df['order_id'].map(installments)
    df['freight_to_price_ratio'] = (df['order_frieght_value_sum']
                                    / df['order_pdt_price_sum'].replace(0, np.nan))

    df['is_weekend_approval'] = approved_dt.dt.dayofweek.ge(5).astype(int)
    df['is_black_friday_period'] = approved_dt.between(*BLACK_FRIDAY_PERIOD).astype(int)
    df['is_december_peak'] = approved_dt.dt.month.eq(12).astype(int)

    # Count approvals over all orders (not only the modelling population) for the 7
    # completed days before each approval day; the approval day itself is excluded.
    all_approvals = olist['orders']['order_approved_at'].dropna().dt.normalize()
    daily = all_approvals.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq='D'),
                          fill_value=0)
    prev_7d = daily.rolling(7, min_periods=1).sum().shift(1).fillna(0)
    df['orders_approved_prev_7d'] = approved_dt.map(prev_7d)
    return df
