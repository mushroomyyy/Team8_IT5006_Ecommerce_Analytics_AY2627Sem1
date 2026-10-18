"""Validate approval-time inputs before applying the fitted model pipeline."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


def validate_features(features: dict[str, Any], metadata: dict[str, Any], model=None) -> pd.DataFrame:
    """Return one typed row; explicit nulls use the pipeline's fitted imputers.

    Unseen categorical strings are accepted: the fitted encoder's unknown-category
    policy applies. Invalid numeric text is never silently converted to missing.
    """
    numeric = metadata["features"]["numeric"]
    categorical = metadata["features"]["categorical"]
    columns = numeric + categorical
    missing = [name for name in columns if name not in features]
    extra = sorted(set(features) - set(columns))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unknown features: {', '.join(extra)}")

    calendar_bounds = {
        "order_approved_day_of_week": (1, 7),
        "order_approved_day_of_month": (1, 31),
        "order_approved_month": (1, 12),
        "order_approved_week_of_year": (1, 53),
    }
    row = {}
    for name in numeric:
        value = features[name]
        if value is None:
            row[name] = np.nan
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number or null")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number or null")
        if name in calendar_bounds:
            low, high = calendar_bounds[name]
            if not low <= value <= high or value != int(value):
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        elif name.startswith("is_"):
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        elif name != "promised_lead_days" and value < 0:
            raise ValueError(f"{name} must be nonnegative")
        if (name.endswith("_count") or name in ("payment_installments_max", "orders_approved_prev_7d")) and value != int(value):
            raise ValueError(f"{name} must be an integer")
        row[name] = float(value)
    if all(pd.isna(row[name]) for name in numeric):
        raise ValueError("At least one numeric model feature must be supplied")
    for name in categorical:
        value = features[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string or null")
        row[name] = np.nan if value is None else value
    frame = pd.DataFrame([row], columns=columns)
    # An all-null categorical column must remain categorical input to sklearn.
    # Otherwise pandas infers float and OneHotEncoder can fail on np.isnan.
    frame[categorical] = frame[categorical].astype(object)
    return frame
