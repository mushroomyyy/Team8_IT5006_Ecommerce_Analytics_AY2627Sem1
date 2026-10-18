"""Strict approval-time feature contract shared by every inference endpoint."""
from __future__ import annotations

import calendar
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

# A domain sanity ceiling, deliberately far below sklearn's float32 overflow limit.
NUMERIC_SANITY_LIMIT = 1e12


class FeatureValidationError(ValueError):
    """A client supplied an invalid engineered feature row."""


def validate_features(features: dict[str, Any], metadata: dict[str, Any], model=None) -> pd.DataFrame:
    """Reject unsupported categories; numeric nulls use fitted median imputation."""
    numeric = metadata["features"]["numeric"]
    categorical = metadata["features"]["categorical"]
    columns = numeric + categorical
    if not isinstance(features, dict):
        raise FeatureValidationError("features must be an object")
    missing = [name for name in columns if name not in features]
    extra = sorted(set(features) - set(columns))
    if missing:
        raise FeatureValidationError(f"Missing required features: {', '.join(missing)}")
    if extra:
        raise FeatureValidationError(f"Unknown features: {', '.join(extra)}")
    calendar_bounds = {"order_approved_day_of_week": (1, 7), "order_approved_day_of_month": (1, 31),
                       "order_approved_month": (1, 12), "order_approved_week_of_year": (1, 53)}
    counts = {"order_item_count", "order_seller_count", "order_product_category_count", "seller_state_count", "payment_installments_max"}
    row = {}
    for name in numeric:
        value = features[name]
        if value is None:
            row[name] = np.nan
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise FeatureValidationError(f"{name} must be a finite number or null")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise FeatureValidationError(f"{name} must be a finite number or null")
        if abs(value) > NUMERIC_SANITY_LIMIT:
            raise FeatureValidationError(f"{name} magnitude must not exceed {NUMERIC_SANITY_LIMIT:g}")
        if name in calendar_bounds:
            low, high = calendar_bounds[name]
            if not low <= value <= high or value != int(value):
                raise FeatureValidationError(f"{name} must be an integer from {low} to {high}")
        elif name.startswith("is_"):
            if value not in (0, 1):
                raise FeatureValidationError(f"{name} must be 0 or 1")
        elif name in counts:
            if value < 1 or value != int(value):
                raise FeatureValidationError(f"{name} must be an integer >= 1")
        elif name == "orders_approved_prev_7d":
            if value < 0 or value != int(value):
                raise FeatureValidationError(f"{name} must be an integer >= 0")
        elif name == "promised_lead_days":
            if value != int(value):
                raise FeatureValidationError(f"{name} must be an integer or null")
        elif value < 0:
            raise FeatureValidationError(f"{name} must be nonnegative")
        row[name] = float(value)
    if all(pd.isna(row[name]) for name in numeric):
        raise FeatureValidationError("At least one numeric model feature must be supplied")
    month, day = features.get("order_approved_month"), features.get("order_approved_day_of_month")
    # Without a year, allow February 29 but never February 30/31.
    if month is not None and day is not None and day > calendar.monthrange(2000, int(month))[1]:
        raise FeatureValidationError("order_approved_day_of_month is invalid for order_approved_month")
    for source, flag, expected in (("order_approved_day_of_week", "is_weekend_approval", lambda x: int(x >= 6)),
                                   ("order_approved_month", "is_december_peak", lambda x: int(x == 12))):
        if features.get(source) is not None and features.get(flag) is not None and features[flag] != expected(features[source]):
            raise FeatureValidationError(f"{flag} is inconsistent with {source}")
    for name in categorical:
        value = features[name]
        allowed = metadata["categories"][name]
        if not isinstance(value, str) or value not in allowed:
            raise FeatureValidationError(f"{name} must exactly match one of: {', '.join(allowed)}; null is not supported")
        row[name] = value
    frame = pd.DataFrame([row], columns=columns)
    frame[categorical] = frame[categorical].astype(object)
    return frame
