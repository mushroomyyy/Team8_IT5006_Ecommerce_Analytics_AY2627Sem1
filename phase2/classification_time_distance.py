"""Purchase-time / static geographic feature ablation on development folds only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from classification_experiments import (ROOT, EXTRA_NUM, EXTRA_CAT, enriched_features,
    make_model, model_input)
from classification_backtest import evaluate_out_of_fold_predictions, fold_metadata
from model_workbench import (NUMERIC_FEATURES, load_order_tables, read_csv_from_archive,
    build_order_dataset, build_temporal_folds, to_builtin)

TIME_FEATURES = ['purchase_weekday', 'purchase_hour', 'purchase_month', 'purchase_weekend',
    'purchase_weekday_sin', 'purchase_weekday_cos', 'purchase_hour_sin', 'purchase_hour_cos',
    'purchase_month_sin', 'purchase_month_cos']
DISTANCE_FEATURES = ['seller_distance_mean_km', 'seller_distance_max_km',
    'seller_distance_missing_fraction', 'log_distance_mean', 'distance_per_promised_day']


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [np.radians(x) for x in [lat1, lon1, lat2, lon2]]
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geographic_lookup(geo):
    # Fixed, outcome-free reference lookup; not a learned target aggregate.
    # Broad Brazil bounding box removes impossible coordinate records.
    valid = geo.loc[geo.geolocation_lat.between(-35, 6) &
                    geo.geolocation_lng.between(-75, -30)]
    return valid.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].median()


def purchase_distance_features(tables, geo, as_of_timestamp=None):
    frame = enriched_features(tables, as_of_timestamp)
    orders = tables['orders'][['order_id', 'customer_id', 'order_purchase_timestamp']].copy()
    stamp = orders.order_purchase_timestamp
    orders['purchase_weekday'], orders['purchase_hour'], orders['purchase_month'] = stamp.dt.dayofweek, stamp.dt.hour, stamp.dt.month
    orders['purchase_weekend'] = (stamp.dt.dayofweek >= 5).astype(float).where(stamp.notna())
    for name, period in [('purchase_weekday', 7), ('purchase_hour', 24), ('purchase_month', 12)]:
        orders[name + '_sin'] = np.sin(2 * np.pi * orders[name] / period)
        orders[name + '_cos'] = np.cos(2 * np.pi * orders[name] / period)
    frame = frame.merge(orders[['order_id', *TIME_FEATURES]], validate='one_to_one')
    lookup = geographic_lookup(geo)
    customers = tables['customers'][['customer_id', 'customer_zip_code_prefix']].merge(
        lookup, left_on='customer_zip_code_prefix', right_index=True, how='left', validate='many_to_one')
    customers = customers.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lon'})
    sellers = tables['sellers'][['seller_id', 'seller_zip_code_prefix']].merge(
        lookup, left_on='seller_zip_code_prefix', right_index=True, how='left', validate='many_to_one')
    # Unique sellers, not item-weighted distances; one order can have several origins.
    routes = tables['items'][['order_id', 'seller_id']].drop_duplicates().merge(
        orders[['order_id', 'customer_id']], validate='many_to_one').merge(
        customers, on='customer_id', how='left', validate='many_to_one').merge(
        sellers, on='seller_id', how='left', validate='many_to_one')
    routes['distance'] = haversine_km(routes.customer_lat, routes.customer_lon,
                                     routes.geolocation_lat, routes.geolocation_lng)
    routes['distance_missing'] = routes.distance.isna().astype(float)
    aggregated = routes.groupby('order_id').agg(seller_distance_mean_km=('distance', 'mean'),
        seller_distance_max_km=('distance', 'max'),
        seller_distance_missing_fraction=('distance_missing', 'mean'))
    frame = frame.merge(aggregated, left_on='order_id', right_index=True, how='left', validate='one_to_one')
    frame['seller_distance_missing_fraction'] = frame.seller_distance_missing_fraction.fillna(1.)
    frame['log_distance_mean'] = np.log1p(frame.seller_distance_mean_km)
    frame['distance_per_promised_day'] = frame.seller_distance_max_km / frame.promised_lead_days.where(frame.promised_lead_days > 0)
    return frame


def run():
    archive = ROOT.parent / 'streamlit_release/data/olist_csv.zip'
    tables = load_order_tables(archive)
    geo = read_csv_from_archive(archive, 'olist_geolocation_dataset.csv')
    feature_frame = purchase_distance_features(tables, geo)
    additions = EXTRA_NUM + EXTRA_CAT + TIME_FEATURES + DISTANCE_FEATURES
    dataset = build_order_dataset(archive).merge(feature_frame[['order_id', *additions]], validate='one_to_one')
    folds = build_temporal_folds(dataset)
    saved = pd.read_csv(ROOT / 'classification_experiment_predictions.csv', parse_dates=['approval_date', 'label_available_at'])
    old = saved.loc[saved.model.str.startswith('enriched_')].copy()
    expected = pd.concat([f['validation'] for f in folds]).set_index('order_id')
    results = dict(status='development_only_purchase_distance_ablation', final_holdout_evaluated=False,
        time_features=TIME_FEATURES, distance_features=DISTANCE_FEATURES, folds=fold_metadata(folds),
        caveats=['Static full-archive postcode coordinate lookup assumed available at deployment; no geographic revision timestamps.',
                 'Straight-line postcode-centroid proxy, not road/carrier travel distance.',
                 'Original eventual-delivery population and snapshot limitations remain.'], models={})
    for name, pred in old.groupby('model'):
        assert pred.order_id.is_unique and set(pred.order_id) == set(expected.index)
        for col, target in [('actual', 'late_delivery'), ('approval_date', 'approval_date')]:
            assert np.array_equal(pred[col].to_numpy(), expected.loc[pred.order_id, target].to_numpy())
        results['models'][name] = evaluate_out_of_fold_predictions(pred.reset_index(drop=True))
    assert len(results['models']) == 4
    predictions = [old]
    for prefix, new_cols in [('purchase', TIME_FEATURES), ('purchase_distance', TIME_FEATURES + DISTANCE_FEATURES)]:
        for name in ['logistic_regression', 'random_forest', 'xgboost', 'catboost']:
            variant = prefix + '_' + name
            parts = []
            for fold in folds:
                print(variant, 'fold', fold['fold'], flush=True)
                train, val = fold['train'], fold['validation']
                y = train.late_delivery
                model = make_model(name, True, y.eq(0).sum() / y.eq(1).sum())
                if name != 'catboost':
                    prep = model.named_steps['preprocess']
                    prep.transformers = [(n, t, NUMERIC_FEATURES + EXTRA_NUM + new_cols if n == 'numeric' else cols)
                                         for n, t, cols in prep.transformers]
                x_train = pd.concat([model_input(train, name, True), train[new_cols]], axis=1)
                x_val = pd.concat([model_input(val, name, True), val[new_cols]], axis=1)
                model.fit(x_train, y)
                pred = val[['order_id', 'approval_date', 'label_available_at']].copy()
                pred['actual'], pred['probability'] = val.late_delivery, model.predict_proba(x_val)[:, 1]
                pred['model'], pred['fold'], pred['sample'] = variant, fold['fold'], 'cross_validation'
                parts.append(pred)
            pred = pd.concat(parts, ignore_index=True)
            results['models'][variant] = evaluate_out_of_fold_predictions(pred)
            predictions.append(pred)
            (ROOT / 'classification_time_distance_results.json').write_text(json.dumps(to_builtin(results), indent=2, allow_nan=False))
            pd.concat(predictions, ignore_index=True).to_csv(ROOT / 'classification_time_distance_predictions.csv', index=False)
    result_rows = dataset.loc[dataset.order_id.isin(expected.index)]
    results['development_distance_audit'] = dict(rows=len(result_rows),
        orders_with_any_missing_seller_distance=int(result_rows.seller_distance_missing_fraction.gt(0).sum()),
        median_mean_distance_km=float(result_rows.seller_distance_mean_km.median()),
        p99_max_distance_km=float(result_rows.seller_distance_max_km.quantile(.99)))
    (ROOT / 'classification_time_distance_results.json').write_text(json.dumps(to_builtin(results), indent=2, allow_nan=False))


if __name__ == '__main__':
    run()
