"""Moving-window backtest for the confirmed late-delivery classifier.

This is the improved evaluation path. ``model_workbench.py`` remains a labelled
historical pilot so its existing results stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_workbench import (
    FEATURES,
    LABEL_GAP_DAYS,
    RANDOM_STATE,
    TRAIN_WINDOW_DAYS,
    build_final_split,
    build_order_dataset,
    build_temporal_folds,
    observable_training_data,
    classification_metrics,
    classification_models,
    select_f1_threshold,
    to_builtin,
)


def summarise_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    metric_names = fold_metrics[0].keys()
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "standard_deviation": float(np.nanstd(values, ddof=1)),
            "minimum": float(np.nanmin(values)),
            "maximum": float(np.nanmax(values)),
        }
    return summary


def fold_metadata(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fold": fold["fold"],
            "training_window_days": fold["training_window_days"],
            "label_gap_days": fold["label_gap_days"],
            "train_rows": len(fold["train"]),
            "train_approval_start": str(
                fold["train"]["approval_date"].min().date()
            ),
            "train_approval_end": str(
                fold["train"]["approval_date"].max().date()
            ),
            "train_late_rate": float(fold["train"]["late_delivery"].mean()),
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "validation_rows": len(fold["validation"]),
            "validation_late_rate": float(
                fold["validation"]["late_delivery"].mean()
            ),
        }
        for fold in folds
    ]


def fit_fold_predictions(
    folds: list[dict[str, Any]], model_name: str
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for fold in folds:
        train = fold["train"]
        validation = fold["validation"]
        y_train = train["late_delivery"]
        scale_pos_weight = float(y_train.eq(0).sum() / y_train.eq(1).sum())
        model = classification_models(scale_pos_weight)[model_name]
        model.fit(train[FEATURES], y_train)
        probability = model.predict_proba(validation[FEATURES])[:, 1]
        predictions.append(
            pd.DataFrame(
                {
                    "order_id": validation["order_id"].to_numpy(),
                    "approval_date": validation["approval_date"].to_numpy(),
                    "actual": validation["late_delivery"].to_numpy(),
                    "label_available_at": validation["label_available_at"].to_numpy(),
                    "probability": probability,
                    "fold": fold["fold"],
                    "model": model_name,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True)


def evaluate_out_of_fold_predictions(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    predictions["predicted"] = 0
    threshold_audit = []
    metrics_by_fold: list[dict[str, float]] = []
    for fold_number, fold_predictions in predictions.groupby("fold", sort=True):
        origin = fold_predictions.approval_date.min().normalize().replace(day=1)
        history = predictions.loc[(predictions.fold < fold_number) &
                                  predictions.label_available_at.le(origin)]
        threshold = (select_f1_threshold(history.actual, history.probability.to_numpy())
                     if history.actual.nunique() == 2 else 0.5)
        predictions.loc[fold_predictions.index, "predicted"] = (
            fold_predictions.probability >= threshold).astype(int)
        threshold_audit.append(dict(fold=int(fold_number), origin=str(origin),
                                    threshold=threshold, history_rows=len(history),
                                    latest_label=str(history.label_available_at.max()) if len(history) else None))
        metrics_by_fold.append(
            classification_metrics(
                fold_predictions["actual"],
                fold_predictions["probability"].to_numpy(),
                threshold,
                dates=fold_predictions.approval_date,
            )
        )
    eligible = predictions.loc[predictions.label_available_at.le(pd.Timestamp("2018-05-01"))]
    threshold = select_f1_threshold(eligible.actual, eligible.probability.to_numpy())
    pooled_metrics = classification_metrics(
        predictions["actual"],
        predictions["probability"].to_numpy(),
        threshold,
        dates=predictions.approval_date,
    )
    # Report prospective threshold decisions, not F1 fitted on the same labels.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    pooled_metrics.pop("threshold")
    pooled_metrics["accuracy"] = accuracy_score(predictions.actual, predictions.predicted)
    for key, metric in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        pooled_metrics[key] = metric(predictions.actual, predictions.predicted, zero_division=0)
    return {
        "threshold_for_exploratory_from_labels_available_by_may_1": threshold,
        "threshold_source_rows": len(eligible),
        "threshold_audit": threshold_audit,
        "pooled_oof_metrics": pooled_metrics,
        "fold_metrics": metrics_by_fold,
        "fold_summary": summarise_fold_metrics(metrics_by_fold),
    }


def fit_exploratory_period(
    final_split: dict[str, pd.DataFrame],
    model_name: str,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    train = final_split["train"]
    test = final_split["test"]
    y_train = train["late_delivery"]
    scale_pos_weight = float(y_train.eq(0).sum() / y_train.eq(1).sum())
    model = classification_models(scale_pos_weight)[model_name]
    model.fit(train[FEATURES], y_train)
    probability = model.predict_proba(test[FEATURES])[:, 1]
    metrics = classification_metrics(
        test["late_delivery"], probability, threshold, dates=test.approval_date
    )
    predictions = pd.DataFrame(
        {
            "order_id": test["order_id"].to_numpy(),
            "approval_date": test["approval_date"].to_numpy(),
            "actual": test["late_delivery"].to_numpy(),
            "probability": probability,
            "model": model_name,
            "label_available_at": test.label_available_at.to_numpy(),
            "predicted": (probability >= threshold).astype(int),
        }
    )
    return metrics, predictions


def run_backtest(dataset: pd.DataFrame, include_exploratory=False) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = build_temporal_folds(dataset)
    final_split = build_final_split(dataset)
    model_names = list(
        classification_models(scale_pos_weight=1.0).keys()
    )

    model_results: dict[str, Any] = {}
    all_predictions: list[pd.DataFrame] = []
    for model_name in model_names:
        print(f"Running six moving-window folds: {model_name}", flush=True)
        oof_predictions = fit_fold_predictions(folds, model_name)
        oof_evaluation = evaluate_out_of_fold_predictions(oof_predictions)
        model_results[model_name] = {
            "cross_validation": oof_evaluation,
        }
        oof_predictions["sample"] = "cross_validation"
        all_predictions.append(oof_predictions)
        if include_exploratory:
            # Monthly refits match CV cadence; threshold stays frozen at May 1.
            period_predictions = []
            for start, end in [("2018-05-01", "2018-05-31"), ("2018-06-01", "2018-06-30")]:
                split = {"train": observable_training_data(dataset, pd.Timestamp(start)),
                         "test": dataset.loc[dataset.approval_date.between(start, end)].copy()}
                _, pred = fit_exploratory_period(split, model_name,
                    oof_evaluation["threshold_for_exploratory_from_labels_available_by_may_1"])
                period_predictions.append(pred)
            test_predictions = pd.concat(period_predictions, ignore_index=True)
            model_results[model_name]["exploratory"] = classification_metrics(
                test_predictions.actual, test_predictions.probability.to_numpy(),
                oof_evaluation["threshold_for_exploratory_from_labels_available_by_may_1"],
                dates=test_predictions.approval_date)
            test_predictions["sample"] = "exploratory"
            all_predictions.append(test_predictions)

    final_train = final_split["train"]
    final_test = final_split["test"]
    results = {
        "status": "corrected_classification_pilot_not_final",
        "final_holdout_evaluated": False,
        "cadence": "Monthly refit; daily scoring and capacity selection",
        "versions": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost"]},
        "random_state": RANDOM_STATE,
        "training_window_days": TRAIN_WINDOW_DAYS,
        "label_gap_days": LABEL_GAP_DAYS,
        "features": FEATURES,
        "folds": fold_metadata(folds),
        "exploratory_period_metadata": {
            "train_rows": len(final_train),
            "train_approval_start": str(
                final_train["approval_date"].min().date()
            ),
            "train_approval_end": str(
                final_train["approval_date"].max().date()
            ),
            "train_late_rate": float(final_train["late_delivery"].mean()),
            "test_rows": len(final_test),
            "test_approval_start": str(
                final_test["approval_date"].min().date()
            ),
            "test_approval_end": str(
                final_test["approval_date"].max().date()
            ),
            "test_late_rate": float(final_test["late_delivery"].mean()),
        },
        "models": model_results,
        "caveats": [
            "Hyperparameters are starting values and have not been tuned.",
            "Monthly validation blocks approximate the proposed daily scoring process.",
            "The 180-day window and 30-day label gap are provisional.",
            "Statistical comparison and calibration are not yet implemented; no significance claims.",
            "May-June was previously inspected and is exploratory only; final evaluation is not implemented.",
            "Pilot population is eventual deliveries; results are conditional and not validated for cancelled or incomplete orders.",
            "All labels mature at end of promised day; overdue pending deliveries are retained as positive.",
            "Related-table availability at approval is assumed because Olist lacks revision histories.",
            "Coverage is expected capture under random cutoff ties; lift compares with random selection at identical daily capacities.",
        ],
    }
    return results, pd.concat(all_predictions, ignore_index=True)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-exploratory", action="store_true", help="Explicitly rerun previously inspected May-June; not a final holdout")
    parser.add_argument(
        "--archive",
        type=Path,
        default=project_root / "streamlit_release" / "data" / "olist_csv.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "phase2"
        / "classification_backtest_results.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root
        / "phase2"
        / "classification_backtest_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_order_dataset(args.archive)
    results, predictions = run_backtest(dataset, args.include_exploratory)
    args.output.write_text(
        json.dumps(to_builtin(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # CSV is ignored by the repository and retained locally for paired testing.
    predictions.to_csv(args.predictions, index=False)
    print(json.dumps(results["exploratory_period_metadata"], indent=2))
    print(f"Wrote results to {args.output}")
    print(f"Wrote paired predictions to {args.predictions}")


if __name__ == "__main__":
    main()
