from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_selection import encoded_block_summary, numeric_vif, stage_summary
from src.features import SELECTED_NUM_COLS, add_extra_features
from src.preprocessing import make_preprocessor


def test_payment_combination_preserves_multi_method_and_missing_orders():
    orders = pd.DataFrame({
        'order_id': ['mixed', 'card', 'none'],
        'order_approved_dt': pd.to_datetime(['2018-01-05'] * 3),
        'order_approved_at': pd.to_datetime(['2018-01-05'] * 3),
        'order_purchase_timestamp': pd.to_datetime(['2018-01-04'] * 3),
        'order_estimated_delivery_date': pd.to_datetime(['2018-01-20'] * 3),
        'order_frieght_value_sum': [3., 2., 1.],
        'order_pdt_price_sum': [10., 8., 5.],
        'payment_value_sum': [13., 10., np.nan],
    })
    products = pd.DataFrame({
        'product_id': ['p'], 'product_length_cm': [2],
        'product_height_cm': [2], 'product_width_cm': [2],
        'product_photos_qty': [1],
    })
    olist = {
        'products': products,
        'order_items': pd.DataFrame({'order_id': ['mixed'], 'product_id': ['p']}),
        'payments': pd.DataFrame({
            'order_id': ['mixed', 'mixed', 'mixed', 'card'],
            'payment_type': ['voucher', 'credit_card', 'voucher', 'credit_card'],
            'payment_installments': [1, 1, 1, 2],
        }),
        'orders': orders[['order_id', 'order_approved_at']],
    }
    result = add_extra_features(orders, olist).set_index('order_id')
    assert result.loc['mixed', 'payment_combination'] == 'credit_card+voucher'
    assert result.loc['card', 'payment_combination'] == 'credit_card'
    assert result.loc['none', 'payment_combination'] == 'no_payment'
    assert result.loc['mixed', 'payment_value_sum'] == 13


def test_payment_combination_uses_credit_card_reference_and_ignores_unknown():
    train = pd.DataFrame({
        'amount': [10., 20., 30., 40.],
        'payment_combination': ['credit_card', 'boleto', 'credit_card+voucher', 'voucher'],
    })
    preprocessor = make_preprocessor(['amount'], ['payment_combination'])
    preprocessor.fit(train)
    assert preprocessor.named_transformers_['payment_combination'].categories_[0].tolist() == [
        'boleto', 'credit_card', 'credit_card+voucher', 'voucher']
    names = preprocessor.get_feature_names_out().tolist()
    assert 'payment_combination__payment_combination_credit_card' not in names
    assert 'payment_combination__payment_combination_credit_card+voucher' in names
    values = preprocessor.transform(pd.DataFrame({
        'amount': [50., 60., 70.],
        'payment_combination': ['credit_card', 'credit_card+voucher', 'new_method'],
    }))
    assert values.shape == (3, 4)
    assert values[0, 1:].sum() == 0
    assert values[1, names.index('payment_combination__payment_combination_credit_card+voucher')] == 1
    assert values[2, 1:].sum() == 0


def test_vif_identifies_linear_dependence_and_independent_features():
    frame = pd.DataFrame({'a': [0., 1., 2., 3., 4.],
                          'b': [1., 0., 1., 0., 1.]})
    assert np.allclose(numeric_vif(frame), 1.)
    frame['sum'] = frame['a'] + frame['b']
    assert np.isinf(numeric_vif(frame)).all()


def test_stage_summary_counts_removals_and_remaining_columns():
    trace = pd.DataFrame({
        'stage': ['constant', 'domain representative', 'domain representative', 'iterative VIF'],
        'removed': ['c', 'd1', 'd2', 'v'],
    })
    summary = stage_summary(trace, start_count=10)
    assert summary['stage'].tolist() == ['start', 'constant', 'domain representative', 'iterative VIF']
    assert summary['removed'].tolist() == [0, 1, 2, 1]
    assert summary['numeric_remaining'].tolist() == [10, 9, 7, 6]
    assert summary.loc[2, 'removed_columns'] == 'd1, d2'


def test_encoded_block_summary_reads_fitted_reference_coding():
    train = pd.DataFrame({
        'amount': [1., 2., np.nan, 4.],
        'customer_state': ['SP', 'RJ', 'MG', 'SP'],
        'route_type': ['1. All interstate', '2. All same-state', '1. All interstate', '3. Mixed'],
        'payment_combination': ['credit_card', 'boleto', 'voucher', 'credit_card'],
    })
    preprocessor = make_preprocessor(['amount'], ['customer_state', 'route_type', 'payment_combination'],
                                     reference_categories=True).fit(train)
    summary = encoded_block_summary(preprocessor).set_index('block')
    assert summary.index.tolist() == ['numeric', 'customer_state', 'route_type',
                                      'payment_combination', 'total']
    assert summary['levels_seen'].drop(['numeric', 'total']).tolist() == [3, 3, 3]
    assert summary['reference_dropped'].drop(['numeric', 'total']).tolist() == [
        'SP', '1. All interstate', 'credit_card']
    assert summary['encoded_columns'].tolist() == [1, 2, 2, 2, 7]
    assert summary.loc['total', 'raw_columns'] == 4
    assert summary.loc['total', 'encoded_columns'] == preprocessor.transform(train).shape[1]


def test_selected_numeric_columns_match_saved_audit():
    saved = pd.read_csv(Path(__file__).parent / 'results/feature_selection/selected_numeric_features.csv')
    assert SELECTED_NUM_COLS == saved['feature'].tolist()
