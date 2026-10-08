import pandas as pd

from src.features import HISTORY_NUM_COLS, add_historical_performance


def test_history_uses_only_outcomes_known_before_approval():
    orders = pd.DataFrame({
        'order_id': ['late', 'early', 'known', 'new', 'no_items'],
        'order_approved_at': pd.to_datetime(['2018-01-01', '2018-01-03',
                                             '2018-01-04', '2018-01-05',
                                             '2018-01-06']),
        'order_estimated_delivery_date': pd.to_datetime([
            '2018-01-03', '2018-01-10', '2018-01-10', '2018-01-10',
            '2018-01-10']),
        'order_delivered_customer_date': pd.to_datetime([
            '2018-01-05', None, None, None, None]),
    })
    items = pd.DataFrame({
        'order_id': ['late', 'early', 'known', 'new'],
        'seller_id': ['s1', 's1', 's1', 's2'],
        'product_id': ['p1', 'p1', 'p1', 'p2'],
    })
    features = orders[['order_id']].copy()
    features['order_approved_dt'] = orders['order_approved_at']
    result = add_historical_performance(features, {'orders': orders, 'order_items': items})
    by_id = result.set_index('order_id')

    assert by_id.loc['early', 'seller_no_history_share'] == 1
    assert by_id.loc['known', 'seller_no_history_share'] == 0
    assert by_id.loc['known', 'product_no_history_share'] == 0
    assert by_id.loc['known', 'seller_late_rate_mean'] == 1
    assert by_id.loc['new', 'seller_no_history_share'] == 1
    assert by_id.loc['new', 'seller_late_rate_mean'] == 1  # as-of global prior
    assert by_id.loc['no_items', 'seller_no_history_share'] == 1
    assert by_id.loc['no_items', 'product_no_history_share'] == 1
    assert not result[HISTORY_NUM_COLS].isna().any().any()

    # A later delivery revision cannot change features for an earlier order.
    changed = orders.copy()
    changed.loc[changed.order_id.eq('late'), 'order_delivered_customer_date'] = pd.Timestamp('2018-01-09')
    revised = add_historical_performance(features, {'orders': changed, 'order_items': items})
    pd.testing.assert_series_equal(
        result.set_index('order_id').loc['early', HISTORY_NUM_COLS],
        revised.set_index('order_id').loc['early', HISTORY_NUM_COLS])

    excluded = add_historical_performance(
        features, {'orders': orders, 'order_items': items},
        exclude_order_ids={'late'})
    assert excluded.set_index('order_id').loc['known', 'seller_no_history_share'] == 1
