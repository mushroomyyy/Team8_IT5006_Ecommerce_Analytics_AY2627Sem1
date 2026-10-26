"""Readable names for the model codes: the single source of truth for every table and figure.

The codes (C0, C1, ..., R3d) stay the stable keys in code and exports; each also has a `short`
label for figure legends and table columns and a longer `description` for the model key.
"""
import pandas as pd

from . import LOOKBACK

MODEL_LABELS = {
    'C0': ('No-skill baseline', 'Predicts the Train late rate for every order'),
    'C1': ('Logistic (all)', 'Plain logistic regression, all 24 features, pre-specified transforms'),
    'C2': ('Logistic (CV-stepwise)', 'Plain logistic, features chosen by CV forward stepwise with the 1-SE rule'),
    'C2B': ('Logistic (BIC)', 'Plain logistic, features chosen by BIC forward stepwise'),
    'C3': ('Random Forest (tuned)',
           'Random Forest, raw 24 features, hyperparameters tuned by two-stage CV search'),
    'C3d': ('Random Forest (default)', 'Random Forest, raw 24 features, scikit-learn default hyperparameters'),
    'C4': ('Decision tree (tuned)', 'Single decision tree, raw 24 features, depth, leaf size and split rule tuned by CV grid search'),
    'R0a': ('Median baseline', 'Predicts the Train median days from promise for every order'),
    'R0b': ('Olist promise',
            f'Predicts arrival exactly on the promised date (0 days), with the lead time capped at {LOOKBACK} days'),
    'R1': ('OLS (all)', 'Ordinary least squares, all 24 features, pre-specified transforms'),
    'R2': ('OLS (CV-stepwise)', 'OLS, features chosen by CV forward stepwise with the 1-SE rule'),
    'R2B': ('OLS (BIC)', 'OLS, features chosen by BIC forward stepwise'),
    'R3': ('Random Forest (tuned)',
           'Random Forest, raw 24 features, hyperparameters tuned by two-stage CV search on RMSE'),
    'R3d': ('Random Forest (default)', 'Random Forest, raw 24 features, scikit-learn default hyperparameters'),
    'R4': ('Decision tree (tuned)', 'Single decision tree, raw 24 features, depth and leaf size tuned by CV grid search on RMSE'),
}
CLASSIFICATION_CODES = ('C0', 'C1', 'C2', 'C2B', 'C4', 'C3', 'C3d')
REGRESSION_CODES = ('R0a', 'R0b', 'R1', 'R2', 'R2B', 'R4', 'R3', 'R3d')


def short_label(code):
    """Short label for figures and table columns, e.g. 'Logistic (CV-stepwise)'."""
    return MODEL_LABELS[code][0]


def description(code):
    """One-line description of what the model is, for the model key."""
    return MODEL_LABELS[code][1]


def display_name(code):
    """Code and short label for legends and prose, e.g. 'C2 · Logistic (CV-stepwise)'."""
    return f'{code} · {short_label(code)}'


def add_model_columns(df, code_col='model'):
    """Copy of `df` with `Code` and `Model` (short label) as the first two columns.

    `code_col` holds the model codes and is replaced by the two new columns.
    """
    codes = df[code_col]
    out = df.drop(columns=code_col)
    out.insert(0, 'Model', codes.map(short_label))
    out.insert(0, 'Code', codes)
    return out


def label_model_columns(df):
    """Copy of `df` whose model-code column headers become `display_name`; other headers are kept."""
    return df.rename(columns={c: display_name(c) for c in df.columns if c in MODEL_LABELS})


def model_key_table(codes, n_features=None):
    """Code / Model / Description / Features table for `codes`.

    `n_features` maps a code to the number of raw features it uses (from the selected lists
    or `run_metadata.json`); codes without an entry show '-'.
    """
    n_features = n_features or {}
    return pd.DataFrame({'Code': list(codes), 'Model': [short_label(c) for c in codes],
                         'Description': [description(c) for c in codes],
                         'Features': [n_features.get(c, '-') for c in codes]})


# Readable names for raw features and fitted terms, used on effect plots.
FEATURE_LABELS = {
    'promised_lead_days': 'Promised lead time', 'approval_lag_hours': 'Purchase-to-approval lag',
    'orders_approved_prev_7d': 'Prior 7-day approvals', 'seller_dist_km_mean': 'Mean seller distance',
    'order_item_count': 'Item count', 'order_seller_count': 'Seller count',
    'order_frieght_value_sum': 'Freight value', 'order_product_weight_g_sum': 'Product weight',
    'order_product_volume_cm3_sum': 'Product volume', 'order_product_photos_mean': 'Product photos',
    'order_product_category_count': 'Category count', 'seller_state_count': 'Seller state count',
    'payment_value_sum': 'Payment value', 'payment_installments_max': 'Instalments',
    'freight_to_price_ratio': 'Freight-to-price ratio', 'is_weekend_approval': 'Weekend approval',
    'is_black_friday_period': 'Black Friday period', 'is_december_peak': 'December peak',
    'order_approved_month': 'Approval month', 'order_approved_day_of_week': 'Approval weekday',
    'order_approved_day_of_month': 'Approval day of month',
}
STATE_NAMES = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AM': 'Amazonas', 'AP': 'Amapá', 'BA': 'Bahia', 'CE': 'Ceará',
    'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão', 'MG': 'Minas Gerais',
    'MS': 'Mato Grosso do Sul', 'MT': 'Mato Grosso', 'PA': 'Pará', 'PB': 'Paraíba', 'PE': 'Pernambuco',
    'PI': 'Piauí', 'PR': 'Paraná', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RO': 'Rondônia',
    'RR': 'Roraima', 'RS': 'Rio Grande do Sul', 'SC': 'Santa Catarina', 'SE': 'Sergipe', 'SP': 'São Paulo',
    'TO': 'Tocantins',
}


def term_label(term, per_unit=False):
    """Readable name for a fitted term, e.g. 'customer_state_AL' -> 'Alagoas vs São Paulo'.

    `per_unit=True` adds the unit of a per-unit effect (doubling, +1 day, yes vs no);
    per-SD plots leave it off because their axis already says per 1 SD.
    """
    if term.startswith('customer_state_'):
        return f"{STATE_NAMES.get(term[15:], term[15:])} vs São Paulo"
    if term.startswith('route_type_'):
        return f"{term.split('. ', 1)[-1]} vs all interstate"
    if term.startswith('payment_combination_'):
        return f"Payment {term[20:].replace('_', ' ')} vs credit card"
    for suffix, name in (('_sin', 'sine'), ('_cos', 'cosine')):
        if term.endswith(suffix):
            return f"{FEATURE_LABELS.get(term[:-4], term[:-4])} ({name})"
    logged = term.startswith('log_')
    feature = term[4:] if logged else term
    name = FEATURE_LABELS.get(feature, feature.replace('_', ' '))
    if not per_unit:
        return f'{name} (log)' if logged else name
    if logged:
        return f'{name}: doubling'
    if feature.startswith('is_'):
        return f'{name}: yes vs no'
    return f'{name}: +1 day' if feature in ('promised_lead_days',) else f'{name}: +1 unit'

