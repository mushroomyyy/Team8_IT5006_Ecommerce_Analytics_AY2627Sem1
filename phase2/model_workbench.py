"""Reproducible Phase 2 pilot models for late-delivery prediction.

This workbench complements ``model_dev.ipynb`` without changing it. It builds an
order-level dataset using information available when an order is approved,
evaluates classification and regression models on chronological cohorts, and
writes report-ready pilot results to JSON.

The results are exploratory until the team agrees on final targets, features,
date boundaries, cross-validation folds, and hyperparameter search spaces.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


RANDOM_STATE = 42
TRAIN_WINDOW_DAYS = 180
LABEL_GAP_DAYS = 30

NUMERIC_FEATURES = [
    "approval_hour",
    "approval_day_of_week",
    "approval_month",
    "approval_is_weekend",
    "promised_lead_days",
    "order_item_count",
    "order_seller_count",
    "order_price_sum",
    "order_freight_sum",
    "freight_to_price_ratio",
    "order_weight_g_sum",
    "order_volume_cm3_sum",
    "order_category_count",
    "seller_state_count",
    "same_state_any",
    "payment_value_sum",
    "payment_installments_max",
    "payment_type_count",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "primary_seller_state",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def read_csv_from_archive(
    archive_path: Path,
    filename: str,
    *,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Read one Olist CSV directly from the repository's bundled archive."""

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(filename) as source:
            return pd.read_csv(source, parse_dates=parse_dates)


def load_order_tables(archive_path: Path) -> dict[str, pd.DataFrame]:
    """Load raw tables; feature construction does not require outcomes."""
    orders = read_csv_from_archive(
        archive_path,
        "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    customers = read_csv_from_archive(
        archive_path, "olist_customers_dataset.csv"
    )
    items = read_csv_from_archive(
        archive_path, "olist_order_items_dataset.csv"
    )
    products = read_csv_from_archive(
        archive_path, "olist_products_dataset.csv"
    )
    sellers = read_csv_from_archive(
        archive_path, "olist_sellers_dataset.csv"
    )
    payments = read_csv_from_archive(
        archive_path, "olist_order_payments_dataset.csv"
    )

    return dict(orders=orders, customers=customers, items=items,
                products=products, sellers=sellers, payments=payments)


def build_order_features(tables: dict[str, pd.DataFrame], as_of_timestamp=None) -> pd.DataFrame:
    """Outcome-independent features for approved orders.

    For live scoring callers must supply related-table snapshots available at
    the cutoff. Olist has no revision timestamps to reconstruct those snapshots.
    Returned data deliberately exclude statuses and actual delivery timestamps.
    """
    orders = tables["orders"]
    customers, items, products, sellers, payments = (
        tables[name] for name in ("customers", "items", "products", "sellers", "payments")
    )
    columns = ["order_id", "customer_id", "order_approved_at", "order_estimated_delivery_date"]
    delivered = orders.loc[orders.order_approved_at.notna(), columns].copy()
    if as_of_timestamp is not None:
        delivered = delivered.loc[delivered.order_approved_at.lt(pd.Timestamp(as_of_timestamp))].copy()

    delivered["approval_date"] = delivered["order_approved_at"].dt.normalize()
    delivered["approval_hour"] = delivered["order_approved_at"].dt.hour
    delivered["approval_day_of_week"] = delivered["order_approved_at"].dt.dayofweek
    delivered["approval_month"] = delivered["order_approved_at"].dt.month
    delivered["approval_is_weekend"] = (
        delivered["approval_day_of_week"] >= 5
    ).astype(int)
    delivered["promised_lead_days"] = (
        delivered["order_estimated_delivery_date"]
        - delivered["order_approved_at"]
    ).dt.total_seconds() / 86_400

    delivered = delivered.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    product_columns = [
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    item_details = items.merge(
        products[product_columns],
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    item_details["product_volume_cm3"] = (
        item_details["product_length_cm"]
        * item_details["product_height_cm"]
        * item_details["product_width_cm"]
    )

    item_agg = (
        item_details.groupby("order_id", as_index=False)
        .agg(
            order_item_count=("product_id", "size"),
            order_seller_count=("seller_id", "nunique"),
            order_price_sum=("price", "sum"),
            order_freight_sum=("freight_value", "sum"),
            order_weight_g_sum=("product_weight_g", lambda x: x.sum(min_count=1)),
            order_volume_cm3_sum=("product_volume_cm3", lambda x: x.sum(min_count=1)),
            order_category_count=("product_category_name", "nunique"),
            seller_state_count=("seller_state", "nunique"),
            primary_seller_state=("seller_state", "first"),
            seller_states=("seller_state", lambda values: frozenset(values.dropna())),
        )
    )

    payment_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value_sum=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_type_count=("payment_type", "nunique"),
        )
    )

    dataset = delivered.merge(
        item_agg, on="order_id", how="left", validate="one_to_one"
    ).merge(
        payment_agg, on="order_id", how="left", validate="one_to_one"
    )
    dataset["freight_to_price_ratio"] = dataset["order_freight_sum"] / dataset[
        "order_price_sum"
    ].replace(0, np.nan)
    dataset["same_state_any"] = [
        int(customer_state in seller_states)
        if pd.notna(customer_state) and isinstance(seller_states, frozenset)
        else np.nan
        for customer_state, seller_states in zip(
            dataset["customer_state"], dataset["seller_states"]
        )
    ]

    return dataset[["order_id", "approval_date", *FEATURES]].sort_values(
        ["approval_date", "order_id"]
    ).reset_index(drop=True)


def build_order_dataset(archive_path: Path) -> pd.DataFrame:
    """Attach retrospective labels separately; pilot population is eventual deliveries.

    This conditional population does not represent cancellations or all approved
    orders. Its deployment generalisability is not established by this pilot.
    """
    tables = load_order_tables(archive_path)
    orders = tables["orders"]
    labels = orders.loc[orders.order_status.eq("delivered") &
                        orders.order_delivered_customer_date.notna() &
                        orders.order_estimated_delivery_date.notna()].copy()
    labels["late_delivery"] = (
        labels.order_delivered_customer_date.dt.normalize() >
        labels.order_estimated_delivery_date.dt.normalize()
    ).astype(int)
    labels["delivery_lead_days"] = (
        labels.order_delivered_customer_date - labels.order_approved_at
    ).dt.total_seconds() / 86400
    # For date-based promises, lateness is established when the promised day ends.
    # Gate the entire cohort by this known deadline, including overdue pending
    # deliveries, rather than selecting only quickly completed orders.
    labels["label_available_at"] = (
        labels.order_estimated_delivery_date.dt.normalize() + pd.Timedelta(days=1)
    )
    dataset = build_order_features(tables).merge(
        labels[["order_id", "order_delivered_customer_date", "late_delivery",
                "delivery_lead_days", "label_available_at"]],
        on="order_id", how="inner", validate="one_to_one"
    )
    dataset = dataset.loc[
        dataset["approval_date"].ge("2017-01-01")
        & dataset["delivery_lead_days"].ge(0)
    ].copy()

    return dataset.sort_values(["approval_date", "order_id"]).reset_index(drop=True)


CV_WINDOWS = [
    ("2017-11-01", "2017-11-30"),
    ("2017-12-01", "2017-12-31"),
    ("2018-01-01", "2018-01-31"),
    ("2018-02-01", "2018-02-28"),
    ("2018-03-01", "2018-03-31"),
    ("2018-04-01", "2018-04-30"),
]


def observable_training_data(
    dataset: pd.DataFrame, prediction_start: pd.Timestamp
) -> pd.DataFrame:
    """Return a fixed moving window with a label-maturation gap."""

    train_end = prediction_start - pd.Timedelta(days=LABEL_GAP_DAYS)
    train_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    return dataset.loc[
        dataset["approval_date"].ge(train_start)
        & dataset["approval_date"].lt(train_end)
        & dataset["label_available_at"].le(prediction_start)
    ].copy()


def build_temporal_folds(
    dataset: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create moving-window validation folds with mature training labels."""

    folds: list[dict[str, Any]] = []
    for fold_number, (start, end) in enumerate(CV_WINDOWS, start=1):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        train = observable_training_data(dataset, start_ts)
        validation = dataset.loc[
            dataset["approval_date"].ge(start_ts)
            & dataset["approval_date"].le(end_ts)
        ].copy()
        if min(len(train), len(validation)) == 0:
            raise ValueError(f"Temporal fold {fold_number} is empty.")
        folds.append(
            {
                "fold": fold_number,
                "validation_start": start,
                "validation_end": end,
                "training_window_days": TRAIN_WINDOW_DAYS,
                "label_gap_days": LABEL_GAP_DAYS,
                "train": train,
                "validation": validation,
            }
        )
    return folds


def build_final_split(
    dataset: pd.DataFrame,
    *,
    test_start: str = "2018-05-01",
    test_end: str = "2018-06-30",
) -> dict[str, pd.DataFrame]:
    """Legacy exploratory May-June split; already inspected, never a final holdout."""

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    train = observable_training_data(dataset, test_start_ts)
    test = dataset.loc[
        dataset["approval_date"].ge(test_start_ts)
        & dataset["approval_date"].le(test_end_ts)
    ].copy()
    if min(len(train), len(test)) == 0:
        raise ValueError("The final training or test cohort is empty.")
    return {"train": train, "test": test}


def chronological_split(
    dataset: pd.DataFrame,
    *,
    validation_start: str = "2018-03-01",
    test_start: str = "2018-05-01",
    test_end: str = "2018-06-30",
) -> dict[str, pd.DataFrame]:
    """Compatibility split used only by the completed first pilot.

    The next implementation stage should replace this single validation split
    with ``build_temporal_folds`` and reserve ``build_final_split`` for final
    testing, as specified in CLASSIFICATION_IMPLEMENTATION_PLAN.md.
    """

    validation_start_ts = pd.Timestamp(validation_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    train = dataset.loc[
        dataset["approval_date"].lt(validation_start_ts)
        & dataset["order_delivered_customer_date"].lt(validation_start_ts)
    ].copy()
    validation = dataset.loc[
        dataset["approval_date"].ge(validation_start_ts)
        & dataset["approval_date"].lt(test_start_ts)
    ].copy()
    test = dataset.loc[
        dataset["approval_date"].ge(test_start_ts)
        & dataset["approval_date"].le(test_end_ts)
    ].copy()
    return {"train": train, "validation": validation, "test": test}


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def classification_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=25,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def regression_models() -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=8,
                        min_samples_leaf=25,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    xgb.XGBRegressor(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def select_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a validation threshold using F1 as a transparent pilot rule."""

    thresholds = np.unique(np.r_[0.0, probabilities, np.nextafter(1.0, 2.0)])
    # Bound runtime while retaining extreme operating points and low risks.
    if len(thresholds) > 501:
        thresholds = np.unique(np.quantile(thresholds, np.linspace(0, 1, 501)))
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def coverage_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
    dates=None,
) -> dict[str, float]:
    ranking = pd.DataFrame({"actual": np.asarray(y_true), "probability": probabilities,
                            "date": np.asarray(dates) if dates is not None else 0})
    total_positive = float(ranking["actual"].sum())
    results: dict[str, float] = {}

    for fraction in fractions:
        captured = selected = random_captured = 0.0
        for _, day in ranking.groupby("date"):
            count = max(1, math.ceil(len(day) * fraction))
            cutoff = day.probability.nlargest(count).iloc[-1]
            above = day.loc[day.probability.gt(cutoff)]
            tied = day.loc[day.probability.eq(cutoff)]
            # Expected capture under uniform random selection at the cutoff tie.
            captured += above.actual.sum() + (count - len(above)) * tied.actual.mean()
            selected += count
            random_captured += count / len(day) * day.actual.sum()
        coverage = captured / total_positive if total_positive else float("nan")
        results[f"coverage_at_{int(fraction * 100)}pct"] = coverage
        results[f"selected_fraction_at_{int(fraction * 100)}pct"] = selected / len(ranking)
        results[f"lift_at_{int(fraction * 100)}pct"] = (
            captured / random_captured if random_captured else float("nan")
        )

    return results


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    dates=None,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        **coverage_metrics(y_true, probabilities, dates=dates),
    }


def regression_metrics(
    y_true: pd.Series, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "mae_days": mean_absolute_error(y_true, predictions),
        "rmse_days": math.sqrt(mean_squared_error(y_true, predictions)),
        "r_squared": r2_score(y_true, predictions),
    }


def evaluate_models(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]
    X_test = test[FEATURES]

    y_class_train = train["late_delivery"]
    y_class_validation = validation["late_delivery"]
    y_class_test = test["late_delivery"]
    scale_pos_weight = float(
        y_class_train.eq(0).sum() / y_class_train.eq(1).sum()
    )

    classification_results: dict[str, Any] = {}
    for name, model in classification_models(scale_pos_weight).items():
        model.fit(X_train, y_class_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold = select_f1_threshold(
            y_class_validation, validation_probability
        )
        test_probability = model.predict_proba(X_test)[:, 1]
        classification_results[name] = {
            "validation": classification_metrics(
                y_class_validation, validation_probability, threshold
            ),
            "test": classification_metrics(
                y_class_test, test_probability, threshold
            ),
        }

    y_reg_train = train["delivery_lead_days"]
    y_reg_validation = validation["delivery_lead_days"]
    y_reg_test = test["delivery_lead_days"]

    regression_results: dict[str, Any] = {}
    for name, model in regression_models().items():
        model.fit(X_train, y_reg_train)
        validation_prediction = model.predict(X_validation)
        test_prediction = model.predict(X_test)
        regression_results[name] = {
            "validation": regression_metrics(
                y_reg_validation, validation_prediction
            ),
            "test": regression_metrics(y_reg_test, test_prediction),
        }

    split_summary = {
        name: {
            "rows": len(frame),
            "approval_start": str(frame["approval_date"].min().date()),
            "approval_end": str(frame["approval_date"].max().date()),
            "late_rate": float(frame["late_delivery"].mean()),
            "mean_delivery_lead_days": float(
                frame["delivery_lead_days"].mean()
            ),
        }
        for name, frame in splits.items()
    }

    return {
        "status": "pilot_not_final",
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "split_summary": split_summary,
        "classification": classification_results,
        "regression": regression_results,
        "caveats": [
            "Pilot dates and hyperparameters have not been tuned.",
            "Only delivered orders are modelled; deployment treatment of cancelled and unavailable outcomes requires a team decision.",
            "This is one chronological holdout, not the final rolling cross-validation design.",
            "The final test cohort must remain untouched once formal tuning begins.",
        ],
    }


def to_builtin(value: Any) -> Any:
    """Convert NumPy/Pandas scalars into JSON-serialisable values."""

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=project_root / "streamlit_release" / "data" / "olist_csv.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "phase2" / "pilot_model_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_order_dataset(args.archive)
    splits = chronological_split(dataset)
    results = evaluate_models(splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(to_builtin(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["split_summary"], indent=2))
    print(f"Wrote pilot results to {args.output}")


if __name__ == "__main__":
    main()
