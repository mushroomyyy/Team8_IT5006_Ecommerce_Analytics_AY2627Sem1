"""Reproduce the fixed, shared feature selection from the Train rows.

Run with ``python -m src.feature_selection`` from ``phase2``. This diagnostic
reads labels only to define the two eligible training populations. It never
uses outcomes, Validation rows, or model scores to choose a predictor.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .data import load_olist_data
from .features import (CAT_COLS, SELECTED_CAT_COLS, EXTRA_NUM_COLS, NUM_COLS, SELECTED_NUM_COLS,
                       add_extra_features, build_feature_table)
from .labels import labels_as_of, regression_targets_as_of
from .preprocessing import make_preprocessor
from .splits import available_training_rows, chronological_split


OUTPUT = Path('results/feature_selection')
VIF_LIMIT = 5.0
PRESERVED_PAYMENT_FEATURES = ['payment_value_sum']
PAYMENT_COMPONENTS = [
    column for column in NUM_COLS
    if column.startswith(('payment_type_value_', 'payment_type_count_'))
]
# Each group has a prespecified, interpretable representative. Other candidates
# enter the data-driven stage, whose only rule is greatest VIF across both tasks.
REPRESENTATIVE_REMOVALS = {
    'order_revenue_sum': 'exact sum of price and freight; retain component candidates for VIF stage',
    'seller_dist_km_min': 'mean represents seller distance',
    'seller_dist_km_max': 'mean represents seller distance',
    'seller_dist_km_median': 'mean represents seller distance',
    'order_approved_week_of_year': 'month and day represent calendar timing',
    **{column: 'payment total and categorical payment combination represent payment'
       for column in PAYMENT_COMPONENTS},
}


PAYMENT_TYPES = ['boleto', 'credit_card', 'debit_card', 'not_defined', 'voucher']
# Group -> (built from, candidate columns). Mirrors the feature table in MULTICOLLINEARITY_NOTES.md.
CANDIDATE_GROUPS = {
    'Approval calendar': ('order_approved_at', [
        'order_approved_day_of_week', 'order_approved_day_of_month', 'order_approved_month',
        'order_approved_week_of_year']),
    'Basket contents': ('order items + products', [
        'order_item_count', 'order_seller_count', 'order_product_category_count',
        'order_pdt_price_sum', 'order_frieght_value_sum', 'order_revenue_sum',
        'order_product_weight_g_sum']),
    'Seller location': ('sellers + geolocation', [
        'seller_dist_km_max', 'seller_dist_km_min', 'seller_dist_km_median', 'seller_dist_km_mean',
        'seller_zip_code_prefix_count', 'seller_city_count', 'seller_state_count']),
    'Payment': ('payments', ['payment_value_sum']
                + [f'payment_type_{kind}_{name}' for kind in ('value', 'count') for name in PAYMENT_TYPES]),
    'Promise and approval': ('orders', ['promised_lead_days', 'approval_lag_hours']),
    'Product catalogue': ('products', ['order_product_volume_cm3_sum', 'order_product_photos_mean']),
    'Price structure': ('items, payments', ['freight_to_price_ratio', 'payment_installments_max']),
    'Seasonality and load': ('approval date, all orders', [
        'is_weekend_approval', 'is_black_friday_period', 'is_december_peak', 'orders_approved_prev_7d']),
    'Categorical': ('customers, items + sellers', ['customer_state', 'route_type']),
}


def candidate_feature_table():
    """One row per feature group (built from, count, columns), checked against `src.features`."""
    listed = [column for _, columns in CANDIDATE_GROUPS.values() for column in columns]
    assert sorted(listed) == sorted(NUM_COLS + EXTRA_NUM_COLS + CAT_COLS), \
        'Candidate groups differ from the feature lists in src.features'
    return pd.DataFrame([{'group': group, 'built_from': built_from, 'count': len(columns),
                          'columns': ', '.join(columns)}
                         for group, (built_from, columns) in CANDIDATE_GROUPS.items()])


def candidate_feature_status():
    """One row per candidate: its group and whether it is in the final 24 raw features."""
    final = set(SELECTED_NUM_COLS + SELECTED_CAT_COLS)
    return pd.DataFrame([{'group': group, 'feature': column, 'kept': column in final}
                         for group, (_, columns) in CANDIDATE_GROUPS.items() for column in columns])


def numeric_vif(frame):
    """Return per-column VIF; constants and exact linear dependencies are infinite."""
    values = frame.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0)
    constant = scale <= 1e-12
    scaled = np.divide(centered, scale, out=np.zeros_like(centered), where=~constant)
    correlation = scaled.T @ scaled / len(scaled)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    zero = eigenvalues <= max(eigenvalues.max(), 1.0) * 1e-10
    aliased = (np.square(eigenvectors[:, zero]).sum(axis=1) > 1e-8) if zero.any() else np.zeros(len(scale), bool)
    inverse = np.linalg.pinv(correlation, rcond=1e-10, hermitian=True)
    result = np.diag(inverse).copy()
    result[constant | aliased] = np.inf
    return pd.Series(result, index=frame.columns, name='vif')


def training_rows(features):
    """Train rows of both tasks, rebuilt independently of `build_task_data` (sizes are asserted)."""
    window = features[features.order_approved_dt.between('2017-04-18', '2018-04-17')]
    classification = labels_as_of(window, '2018-06-02')
    classification = classification[classification.label_as_of_run.notna()].copy()
    classification['y'] = classification.label_as_of_run.astype(int)
    classification, _ = chronological_split(classification, test_days=30, gap_days=45)
    classification = available_training_rows(
        classification, '2018-03-19', 'y', lambda rows, date: labels_as_of(rows, date),
        'label_as_of_run')
    regression = regression_targets_as_of(
        window[~window.order_status.isin(['canceled', 'unavailable'])], '2018-06-02', 45)
    regression = regression[regression.target_as_of_run.notna()].copy()
    regression['y'] = regression.target_as_of_run.astype(float)
    regression, _ = chronological_split(regression, test_days=30, gap_days=45)
    regression = available_training_rows(
        regression, '2018-03-19', 'y',
        lambda rows, date: regression_targets_as_of(rows, date, 45), 'target_as_of_run')
    assert (len(classification), len(regression)) == (45972, 45371)
    return {'classification': classification, 'regression': regression}


def exact_overlap_table(rows_by_task):
    """Exact-identity audit for each task, as one DataFrame."""
    return pd.DataFrame([{'task': task, **record}
                         for task, rows in rows_by_task.items()
                         for record in exact_overlap_audit(rows)])


def imputed_numeric(rows):
    """Median-imputed candidate numeric columns of one task's Train rows (the VIF input)."""
    columns = NUM_COLS + EXTRA_NUM_COLS
    raw = rows[columns].astype(float)
    matrix = SimpleImputer(strategy='median').fit_transform(raw)
    return pd.DataFrame(matrix, columns=columns, index=rows.index)


def exact_overlap_audit(rows, output=OUTPUT):
    """Record accounting identities and distance equality on eligible rows."""
    records = []
    value_columns = [column for column in NUM_COLS if column.startswith('payment_type_value_')]
    expressions = {
        'revenue_minus_price_and_freight': (
            rows.order_revenue_sum - rows.order_pdt_price_sum - rows.order_frieght_value_sum),
        'payment_total_minus_type_values': (
            rows.payment_value_sum - rows[value_columns].sum(axis=1)),
        'distance_max_minus_mean': rows.seller_dist_km_max - rows.seller_dist_km_mean,
        'distance_mean_minus_min': rows.seller_dist_km_mean - rows.seller_dist_km_min,
    }
    for name, difference in expressions.items():
        valid = difference.dropna().abs()
        records.append({'comparison': name, 'rows': len(valid),
                        'equal_fraction_1e_8': valid.le(1e-8).mean(),
                        'max_absolute_difference': valid.max()})
    return records


def audit(matrices, categorical_rows=None, output=OUTPUT):
    """Save a complete removal trace and before/after diagnostics."""
    output.mkdir(parents=True, exist_ok=True)
    columns = list(next(iter(matrices.values())).columns)
    assert all(list(frame.columns) == columns for frame in matrices.values())
    before = pd.concat([numeric_vif(frame).rename(task) for task, frame in matrices.items()], axis=1)
    before.to_csv(output / 'numeric_vif_before.csv', index_label='feature')
    active = columns.copy()
    trace = []

    def remove(feature, reason, stage):
        scores = {task: numeric_vif(frame[active])[feature] for task, frame in matrices.items()}
        trace.append({'step': len(trace) + 1, 'stage': stage, 'removed': feature,
                      'reason': reason, **{f'{task}_vif_at_removal': value
                                         for task, value in scores.items()}})
        active.remove(feature)

    # Constant features cannot be fed to the unchanged shared preprocessor.
    constants = [column for column in active if any(
        matrices[task][column].nunique() <= 1 for task in matrices)]
    for column in constants:
        remove(column, 'constant in at least one development-training task', 'constant')
    for column in columns:
        if column in active and column in REPRESENTATIVE_REMOVALS:
            remove(column, REPRESENTATIVE_REMOVALS[column], 'domain representative')
    while True:
        scores = pd.concat([numeric_vif(frame[active]).rename(task)
                            for task, frame in matrices.items()], axis=1)
        maximum = scores.max(axis=1)
        worst = maximum.sort_values(ascending=False, kind='stable')
        candidates = worst.drop(index=PRESERVED_PAYMENT_FEATURES)
        if candidates.iloc[0] <= VIF_LIMIT:
            break
        # Keep the payment-value representative; combination is categorical.
        remove(candidates.index[0], 'greatest eligible VIF across tasks; payment representation retained',
               'iterative VIF')
    after = pd.concat([numeric_vif(frame[active]).rename(task)
                       for task, frame in matrices.items()], axis=1)
    after.to_csv(output / 'numeric_vif_after.csv', index_label='feature')
    pd.DataFrame(trace).to_csv(output / 'removal_trace.csv', index=False)
    pd.DataFrame({'feature': active}).to_csv(output / 'selected_numeric_features.csv', index=False)

    if categorical_rows is not None:
        # Inspect the actual optional reference-coded model design, including
        # imputation and all three categorical groups fitted on training rows.
        category_diagnostics = []
        rank_diagnostics = []
        for task, rows in categorical_rows.items():
            preprocessor = make_preprocessor(active, SELECTED_CAT_COLS,
                                             reference_categories=True)
            transformed = preprocessor.fit_transform(rows)
            design = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
            values = numeric_vif(design)
            for feature, value in values.items():
                category_diagnostics.append({'task': task, 'feature': feature, 'vif': value})
            centered = design.to_numpy() - design.to_numpy().mean(axis=0)
            rank = int(np.linalg.matrix_rank(centered, tol=1e-8))
            rank_diagnostics.append({'task': task, 'rows': len(design),
                                     'encoded_columns': design.shape[1],
                                     'centered_rank': rank,
                                     'full_rank_with_intercept': rank == design.shape[1],
                                     'max_encoded_vif': values.max()})
        pd.DataFrame(category_diagnostics).to_csv(output / 'reference_coded_vif.csv', index=False)
        pd.DataFrame(rank_diagnostics).to_csv(output / 'encoded_design_rank.csv', index=False)
    return active


STAGES = ['constant', 'domain representative', 'iterative VIF']


def stage_summary(trace, start_count):
    """Summarise a removal trace as one row per stage, with columns remaining."""
    trace = pd.DataFrame(trace)
    records = [{'stage': 'start', 'removed': 0, 'removed_columns': '',
                'numeric_remaining': start_count}]
    remaining = start_count
    for stage in STAGES:
        removed = trace.loc[trace['stage'].eq(stage), 'removed'].tolist() if len(trace) else []
        remaining -= len(removed)
        records.append({'stage': stage, 'removed': len(removed),
                        'removed_columns': ', '.join(removed), 'numeric_remaining': remaining})
    return pd.DataFrame(records)


def encoded_block_summary(preprocessor):
    """Count levels, dropped references, and encoded columns in a fitted preprocessor."""
    names = preprocessor.get_feature_names_out()
    records = []
    for block, transformer, columns in preprocessor.transformers_:
        if block == 'remainder':
            continue
        if block == 'numeric':
            width = sum(name.startswith('numeric__') for name in names)
            records.append({'block': 'numeric', 'raw_columns': len(columns), 'levels_seen': np.nan,
                            'reference_dropped': '', 'encoded_columns': width})
            continue
        drop = transformer.drop_idx_
        for position, column in enumerate(columns):
            levels = transformer.categories_[position]
            index = None if drop is None else drop[position]
            records.append({'block': column, 'raw_columns': 1, 'levels_seen': len(levels),
                            'reference_dropped': '' if index is None else str(levels[index]),
                            'encoded_columns': len(levels) - (index is not None)})
    summary = pd.DataFrame(records)
    total = {'block': 'total', 'raw_columns': summary['raw_columns'].sum(), 'levels_seen': np.nan,
             'reference_dropped': '', 'encoded_columns': summary['encoded_columns'].sum()}
    assert total['encoded_columns'] == len(names)
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    return summary.astype({'levels_seen': 'Int64'})


if __name__ == '__main__':
    olist = load_olist_data(verbose=False)
    rows = training_rows(add_extra_features(build_feature_table(olist), olist))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exact_overlap_table(rows).to_csv(OUTPUT / 'exact_overlaps.csv', index=False)
    selected = audit({task: imputed_numeric(frame) for task, frame in rows.items()}, rows)
    print(f'{len(selected)} selected numeric features: {selected}')
    print(f'Matches src.features.SELECTED_NUM_COLS: {selected == SELECTED_NUM_COLS}')
    print(f'Selected categorical features: {SELECTED_CAT_COLS}')
