"""Development-only feature ablation; no May-June or final holdout scoring."""
import json
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from model_workbench import (FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    load_order_tables, build_order_features, build_order_dataset, build_temporal_folds,
    classification_models, to_builtin)
from classification_backtest import evaluate_out_of_fold_predictions, fold_metadata

ROOT = Path(__file__).resolve().parent
EXTRA_NUM = ['purchase_to_approval_hours', 'promise_weekday', 'price_per_item',
    'freight_per_item', 'weight_per_item', 'volume_per_item', 'weight_per_volume',
    'price_per_weight', 'multi_seller', 'product_weight_missing',
    'approval_hour_sin', 'approval_hour_cos', 'approval_weekday_sin',
    'approval_weekday_cos', 'approval_month_sin', 'approval_month_cos']
EXTRA_CAT = ['primary_category', 'primary_payment_type', 'state_route']


def enriched_features(tables, as_of_timestamp=None):
    """Only approval-time fields; no delivery/status/review/target aggregates.

    Related-table revision histories are unavailable: snapshot availability
    remains an assumption, as it is for the original features.
    """
    frame = build_order_features(tables, as_of_timestamp)
    orders = tables['orders'][['order_id', 'order_purchase_timestamp',
                              'order_approved_at', 'order_estimated_delivery_date']].copy()
    orders['purchase_to_approval_hours'] = ((orders.order_approved_at -
        orders.order_purchase_timestamp).dt.total_seconds() / 3600).clip(lower=0)
    orders['promise_weekday'] = orders.order_estimated_delivery_date.dt.dayofweek
    frame = frame.merge(orders[['order_id', *EXTRA_NUM[:2]]], validate='one_to_one')
    # Deterministic largest-price item/payment; ties use recorded sequence.
    items = tables['items'].merge(tables['products'][['product_id', 'product_category_name']],
                                 validate='many_to_one', on='product_id')
    primary = items.sort_values(['order_id', 'price', 'order_item_id'],
                                ascending=[True, False, True]).drop_duplicates('order_id')
    frame = frame.merge(primary[['order_id', 'product_category_name']].rename(
        columns={'product_category_name': 'primary_category'}), how='left', validate='one_to_one')
    payments = tables['payments'].sort_values(['order_id', 'payment_value', 'payment_sequential'],
        ascending=[True, False, True]).drop_duplicates('order_id')
    frame = frame.merge(payments[['order_id', 'payment_type']].rename(
        columns={'payment_type': 'primary_payment_type'}), how='left', validate='one_to_one')
    for name, numerator in [('price', 'order_price_sum'), ('freight', 'order_freight_sum'),
                            ('weight', 'order_weight_g_sum'), ('volume', 'order_volume_cm3_sum')]:
        frame[name + '_per_item'] = frame[numerator] / frame.order_item_count.replace(0, np.nan)
    frame['weight_per_volume'] = frame.order_weight_g_sum / frame.order_volume_cm3_sum.replace(0, np.nan)
    frame['price_per_weight'] = frame.order_price_sum / frame.order_weight_g_sum.replace(0, np.nan)
    frame['multi_seller'] = (frame.order_seller_count > 1).astype(int)
    frame['product_weight_missing'] = frame.order_weight_g_sum.isna().astype(int)
    frame['state_route'] = frame.customer_state.fillna('missing') + '->' + frame.primary_seller_state.fillna('missing')
    for prefix, column, period in [('approval_hour', 'approval_hour', 24),
        ('approval_weekday', 'approval_day_of_week', 7), ('approval_month', 'approval_month', 12)]:
        frame[prefix + '_sin'] = np.sin(2 * np.pi * frame[column] / period)
        frame[prefix + '_cos'] = np.cos(2 * np.pi * frame[column] / period)
    return frame.replace([np.inf, -np.inf], np.nan)


def make_model(name, enriched, ratio):
    cats = CATEGORICAL_FEATURES + (EXTRA_CAT if enriched else [])
    if name == 'catboost':
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=.05,
            loss_function='Logloss', auto_class_weights='Balanced', cat_features=cats,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
    model = classification_models(ratio)[name]
    if enriched:
        transformer = model.named_steps['preprocess']
        transformer.transformers = [(n, t, NUMERIC_FEATURES + EXTRA_NUM if n == 'numeric'
                                    else cats) for n, t, _ in transformer.transformers]
    return model


def model_input(frame, name, enriched):
    cols = FEATURES + (EXTRA_NUM + EXTRA_CAT if enriched else [])
    x = frame[cols].copy()
    if name == 'catboost':
        for col in CATEGORICAL_FEATURES + (EXTRA_CAT if enriched else []):
            x[col] = x[col].fillna('missing').astype(str)
    return x


def main():
    archive = ROOT.parent / 'streamlit_release/data/olist_csv.zip'
    dataset = build_order_dataset(archive)
    features = enriched_features(load_order_tables(archive))
    dataset = dataset.merge(features[['order_id', *EXTRA_NUM, *EXTRA_CAT]], validate='one_to_one')
    folds = build_temporal_folds(dataset)
    results = dict(status='development_only_exploratory_ablation', final_holdout_evaluated=False,
        extra_numeric=EXTRA_NUM, extra_categorical=EXTRA_CAT, folds=fold_metadata(folds),
        versions={p: importlib.metadata.version(p) for p in ['catboost', 'scikit-learn', 'xgboost']}, models={})
    saved = pd.read_csv(ROOT / 'classification_backtest_predictions.csv',
                        parse_dates=['approval_date', 'label_available_at'])
    old = saved.loc[saved['sample'].eq('cross_validation') & saved.model.isin(
        ['logistic_regression', 'random_forest', 'xgboost'])].copy()
    # Assert reused baseline predictions match this exact cohort and labels.
    expected = pd.concat([f['validation'] for f in folds]).set_index('order_id')
    for name, pred in old.groupby('model'):
        assert pred.order_id.is_unique and set(pred.order_id) == set(expected.index)
        assert np.array_equal(pred.actual.to_numpy(), expected.loc[pred.order_id, 'late_delivery'].to_numpy())
        assert np.array_equal(pred.approval_date.to_numpy(), expected.loc[pred.order_id, 'approval_date'].to_numpy())
        results['models']['base_' + name] = evaluate_out_of_fold_predictions(pred.reset_index(drop=True))
    old['model'] = 'base_' + old.model
    all_predictions = [old]
    for name, enriched in [('catboost', False), ('logistic_regression', True),
                           ('random_forest', True), ('xgboost', True), ('catboost', True)]:
        variant = ('enriched_' if enriched else 'base_') + name
        predictions = []
        for fold in folds:
            print(variant, 'fold', fold['fold'], flush=True)
            train, val = fold['train'], fold['validation']
            y = train.late_delivery
            model = make_model(name, enriched, y.eq(0).sum() / y.eq(1).sum())
            model.fit(model_input(train, name, enriched), y)
            pred = val[['order_id', 'approval_date', 'label_available_at']].copy()
            pred['actual'] = val.late_delivery
            pred['probability'] = model.predict_proba(model_input(val, name, enriched))[:, 1]
            pred['model'], pred['fold'], pred['sample'] = variant, fold['fold'], 'cross_validation'
            predictions.append(pred)
        pred = pd.concat(predictions, ignore_index=True)
        results['models'][variant] = evaluate_out_of_fold_predictions(pred)
        all_predictions.append(pred)
        (ROOT / 'classification_experiment_results.json').write_text(json.dumps(to_builtin(results), indent=2, allow_nan=False))
        pd.concat(all_predictions, ignore_index=True).to_csv(ROOT / 'classification_experiment_predictions.csv', index=False)


if __name__ == '__main__':
    main()
