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
    'R0a': ('Median baseline', 'Predicts the Train median days from promise for every order'),
    'R0b': ('Olist promise',
            f'Predicts arrival exactly on the promised date (0 days), with the lead time capped at {LOOKBACK} days'),
    'R1': ('OLS (all)', 'Ordinary least squares, all 24 features, pre-specified transforms'),
    'R2': ('OLS (CV-stepwise)', 'OLS, features chosen by CV forward stepwise with the 1-SE rule'),
    'R2B': ('OLS (BIC)', 'OLS, features chosen by BIC forward stepwise'),
    'R3': ('Random Forest (tuned)',
           'Random Forest, raw 24 features, hyperparameters tuned by two-stage CV search on RMSE'),
    'R3d': ('Random Forest (default)', 'Random Forest, raw 24 features, scikit-learn default hyperparameters'),
}
CLASSIFICATION_CODES = ('C0', 'C1', 'C2', 'C2B', 'C3', 'C3d')
REGRESSION_CODES = ('R0a', 'R0b', 'R1', 'R2', 'R2B', 'R3', 'R3d')


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
