"""Transformations used only by the plain linear models (log1p, cyclic calendar, squares).

`make_linear_preprocessor` expands the raw columns first, then reuses
`make_preprocessor` (median imputation, standardisation, reference-coded
categoricals). Everything is fitted on the rows passed to `fit`, so inside a
Pipeline it only ever sees training rows. Trees do not need any of this.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .preprocessing import make_preprocessor

CYCLIC_PERIODS = {'order_approved_month': 12, 'order_approved_day_of_week': 7}
# Every mixed-route order has at least one interstate leg. The level is rare (118 Train
# orders, none late), which makes its plain-logistic coefficient unidentifiable.
ROUTE_MERGES = {'3. Mixed': '1. All interstate'}
CATEGORICAL_PREFIXES = ('numeric__', 'categorical__', 'payment_combination__')


class ExpandColumns(BaseEstimator, TransformerMixin):
    """Apply log1p, sine/cosine and squared terms to the listed raw columns.

    Only columns present in the input are touched, so the same settings work for
    any subset of units. A squared term is centred on the training mean of the
    (possibly logged) column to keep it less collinear with the linear term.
    `merge_mixed_route=True` folds the rare mixed route into all-interstate.
    Output keeps numeric columns first (renamed), then the categorical columns.
    """

    def __init__(self, num_cols, cat_cols, log_cols=(), cyclic=False, squared_cols=(),
                 merge_mixed_route=False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.log_cols = log_cols
        self.cyclic = cyclic
        self.squared_cols = squared_cols
        self.merge_mixed_route = merge_mixed_route

    def _logged(self, X, column):
        values = X[column].astype(float)
        if column in set(self.log_cols):
            if (values.dropna() < 0).any():
                raise ValueError(f'log1p needs non-negative values: {column}')
            values = np.log1p(values)
        return values

    def fit(self, X, y=None):
        self.source_ = {}
        self.output_numeric_ = []
        self.square_centres_ = {}
        log_cols, squared = set(self.log_cols), set(self.squared_cols)
        for column in self.num_cols:
            cyclic = self.cyclic and column in CYCLIC_PERIODS
            if cyclic:
                names = [f'{column}_sin', f'{column}_cos']
            elif column in log_cols:
                names = [f'log_{column}']
            else:
                names = [column]
            if column in squared and not cyclic:
                names.append(f'sq_{column}')
                self.square_centres_[column] = float(self._logged(X, column).mean())
            for name in names:
                self.source_[name] = column
            self.output_numeric_ += names
        for column in self.cat_cols:
            self.source_[column] = column
        return self

    def transform(self, X):
        out = {}
        for column in self.num_cols:
            if self.cyclic and column in CYCLIC_PERIODS:
                angle = 2 * np.pi * X[column].astype(float) / CYCLIC_PERIODS[column]
                out[f'{column}_sin'], out[f'{column}_cos'] = np.sin(angle), np.cos(angle)
                continue
            values = self._logged(X, column)
            out[f'log_{column}' if column in set(self.log_cols) else column] = values
            if column in self.square_centres_:
                out[f'sq_{column}'] = (values - self.square_centres_[column]) ** 2
        for column in self.cat_cols:
            out[column] = X[column]
            if column == 'route_type' and self.merge_mixed_route:
                out[column] = X[column].replace(ROUTE_MERGES)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_numeric_ + list(self.cat_cols), dtype=object)


def make_linear_preprocessor(num_cols, cat_cols, log_cols=(), cyclic=False, squared_cols=(),
                             merge_mixed_route=False):
    """Unfitted Pipeline: expand columns, then impute, scale and reference-code them.

    `log_cols` and `squared_cols` are filtered to `num_cols`, so one setting can be
    reused for every candidate subset during variable selection.
    """
    num_cols, cat_cols = list(num_cols), list(cat_cols)
    log_cols = [c for c in log_cols if c in num_cols]
    squared_cols = [c for c in squared_cols if c in num_cols]
    expand = ExpandColumns(num_cols, cat_cols, log_cols, cyclic, squared_cols, merge_mixed_route)
    # Output names are only known after fitting; numeric names are computed up front here
    numeric_out = []
    for column in num_cols:
        if cyclic and column in CYCLIC_PERIODS:
            numeric_out += [f'{column}_sin', f'{column}_cos']
            continue
        numeric_out.append(f'log_{column}' if column in log_cols else column)
        if column in squared_cols:
            numeric_out.append(f'sq_{column}')
    encode = make_preprocessor(numeric_out, cat_cols, scale=True, reference_categories=True)
    return Pipeline([('expand', expand), ('encode', encode)])


def _encoder(preprocessor):
    """The ColumnTransformer inside a linear preprocessor or a plain make_preprocessor."""
    return preprocessor.named_steps['encode'] if isinstance(preprocessor, Pipeline) else preprocessor


def readable_names(preprocessor):
    """Encoded column names without the ColumnTransformer prefix."""
    names = []
    for name in _encoder(preprocessor).get_feature_names_out():
        for prefix in CATEGORICAL_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        names.append(name)
    return names


def unit_columns(preprocessor, cat_cols):
    """Map each raw feature (unit) to the readable encoded columns it produces.

    A categorical feature maps to its whole dummy block; sine/cosine pairs, log
    versions and squared terms map back to their raw feature.
    """
    source = preprocessor.named_steps['expand'].source_ if isinstance(preprocessor, Pipeline) else {}
    mapping = {}
    for name in readable_names(preprocessor):
        if name in source:
            unit = source[name]
        else:
            # Dummy names start with their raw column name; the longest match wins
            matches = [c for c in cat_cols if name.startswith(f'{c}_')]
            unit = max(matches, key=len) if matches else name
        mapping.setdefault(unit, []).append(name)
    return mapping


def encoded_scales(preprocessor):
    """Standard deviation used to standardise each numeric encoded column (training rows)."""
    encoder = _encoder(preprocessor)
    for block, transformer, columns in encoder.transformers_:
        if block == 'numeric':
            return pd.Series(transformer.named_steps['scaler'].scale_, index=list(columns))
    raise ValueError('Preprocessor has no scaled numeric block')


def linear_design(rows, num_cols, cat_cols, **transform_kwargs):
    """Fit the linear preprocessor on `rows` and return (design DataFrame, unit mapping, fitted).

    The design has readable column names and is the matrix the statsmodels fits use.
    """
    pre = make_linear_preprocessor(num_cols, cat_cols, **transform_kwargs).fit(rows)
    names = readable_names(pre)
    design = pd.DataFrame(pre.transform(rows), columns=names, index=rows.index)
    return design, unit_columns(pre, cat_cols), pre


def skewness_table(train_rows, candidates, limit=1.0):
    """Skewness on Train before and after log1p; log1p recommended when |skew| > limit."""
    records = []
    for column in candidates:
        values = train_rows[column].astype(float).dropna()
        non_negative = bool((values >= 0).all())
        skew_raw = float(values.skew())
        skew_log = float(np.log1p(values).skew()) if non_negative else np.nan
        records.append({'feature': column, 'min': values.min(), 'skew_raw': skew_raw,
                        'skew_log1p': skew_log, 'non_negative': non_negative,
                        'log1p_recommended': non_negative and abs(skew_raw) > limit})
    return pd.DataFrame(records)
