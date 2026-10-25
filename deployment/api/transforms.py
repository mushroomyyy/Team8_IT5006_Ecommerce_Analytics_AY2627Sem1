"""Standalone fitted R1 column transformer.

ExpandColumns is copied without behavioural changes from phase2/src/linear_transforms.py.
The artifact is rebound to this import path without fitting; export verification
compares every June replay prediction exactly against the original artifact.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CYCLIC_PERIODS = {'order_approved_month': 12, 'order_approved_day_of_week': 7}
ROUTE_MERGES = {'3. Mixed': '1. All interstate'}

class ExpandColumns(BaseEstimator, TransformerMixin):
    """Apply log1p and sine/cosine terms to the listed raw columns.

    Only columns present in the input are touched, so the same settings work for
    any subset of units.
    `merge_mixed_route=True` folds the rare mixed route into all-interstate.
    Output keeps numeric columns first (renamed), then the categorical columns.
    """

    def __init__(self, num_cols, cat_cols, log_cols=(), cyclic=False, merge_mixed_route=False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.log_cols = log_cols
        self.cyclic = cyclic
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
        log_cols = set(self.log_cols)
        for column in self.num_cols:
            cyclic = self.cyclic and column in CYCLIC_PERIODS
            if cyclic:
                names = [f'{column}_sin', f'{column}_cos']
            elif column in log_cols:
                names = [f'log_{column}']
            else:
                names = [column]
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
        for column in self.cat_cols:
            out[column] = X[column]
            if column == 'route_type' and self.merge_mixed_route:
                out[column] = X[column].replace(ROUTE_MERGES)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_numeric_ + list(self.cat_cols), dtype=object)
