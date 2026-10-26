"""Export the selected classifier from the repository root with -m deployment.export.train_classification.

The model is the one notebook 02 chose on Validation (`selected_model` in run_metadata.json),
refitted on the known-outcome history, never on June labels. Its operating threshold is the
Validation F1 cut-off recorded by the notebook. Raw Olist data is needed only for this
offline export, not for the deployed API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.export.common import (FAMILY_NAMES, RESULTS, build_spec, fit_selected, notebook_fingerprint,  # noqa: E402
                                      read_csv, rebind_expand_columns, records, save_bundle, selected_model)
from src import RANDOM_STATE  # noqa: E402
from src.data import load_olist_data  # noqa: E402
from src.datasets import EXCLUDED_STATUSES, RUN_DATE, build_task_data  # noqa: E402
from src.evaluation import classification_eval_metrics, score_months  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402
from src.inference import month_bounds, run_inference  # noqa: E402

TASK = "classification"
OUTPUT = RESULTS / TASK
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
THRESHOLD_POLICY = (
    "Operating cut-off: the probability that maximises F1 on the Validation set for the Train-only fit "
    "of the selected model, applied unchanged to the June refit and later months; never chosen on June. "
    "A flag is a screening aid, not a guarantee. Top-10% evaluation ranks an entire cohort and is not "
    "this single-order cut-off."
)
PREPROCESSING = {
    "linear": "Log1p of skewed amounts, sine/cosine calendar terms, median imputation, standardisation and reference-coded one-hot encoding of the selected features; other inputs are accepted and ignored. API rejects null and unknown categorical values before encoding.",
    "tree": "Raw numeric median imputation and reference-coded one-hot encoding. API rejects null and unknown categorical values before encoding.",
    "forest": "Raw numeric median imputation and reference-coded one-hot encoding. API rejects null and unknown categorical values before encoding.",
}


def linear_importance(code: str) -> list[dict]:
    """Rank the selected features of a linear model by their largest absolute standardised effect.

    Numeric, logged and calendar terms are on the standardised scale; a categorical feature
    is scored by its largest absolute level effect against the reference level.
    """
    coefficients = read_csv(OUTPUT / f"{code}_coefficients.csv")
    coefficients = coefficients[coefficients["kind"].ne("intercept")].copy()
    coefficients["importance"] = coefficients["coef"].abs()
    top = coefficients.loc[coefficients.groupby("unit")["importance"].idxmax()].sort_values("importance", ascending=False)
    top["odds_ratio"] = np.exp(top["coef"])
    top["direction"] = np.where(top["coef"] > 0, "raises late risk", "lowers late risk")
    top["rank"] = range(1, len(top) + 1)
    return records(top.rename(columns={"unit": "feature"})[["rank", "feature", "term", "kind", "coef", "odds_ratio", "p_value", "importance", "direction"]])


def export(output_dir: Path = ARTIFACTS) -> dict:
    run, code, family = selected_model(TASK)
    olist = load_olist_data(verbose=False)
    final_df = add_extra_features(build_feature_table(olist), olist)
    data = build_task_data(TASK, final_df)
    if len(data.history_df) != run["row_counts"]["history"]:
        raise ValueError("Refit population no longer matches the Phase 2 run")
    model = fit_selected(TASK, build_spec(TASK, run, code, family, data), family, data)

    # Score exactly the same June eligibility and as-of labels as notebook 02.
    june = score_months(model, final_df, olist["orders"], data.feature_cols, ["2018-06"])["2018-06"]
    reproduced = classification_eval_metrics(
        june["actual_label"], june["predicted_probability"], order_ids=june["order_id"])
    expected_ap = run["headline_june"][code]["avg_precision"]
    if not np.isclose(reproduced["avg_precision"], expected_ap, atol=1e-10, rtol=0):
        raise ValueError(f"June AP {reproduced['avg_precision']} differs from notebook {expected_ap}")
    actual_fingerprint = notebook_fingerprint(TASK, model, family)
    reference_fingerprint = run["fingerprints"][code]
    if actual_fingerprint != reference_fingerprint:
        raise AssertionError(f"Exported {code} fitted-state fingerprint differs from the notebook")

    history_ids = set(data.history_df["order_id"])
    june_rows = final_df.set_index("order_id").loc[june["order_id"]].copy()
    june_rows = june_rows.sort_values(["order_approved_dt", "order_id"])
    # The approval-day worklist cannot know about cancellations, so replay every approved June order.
    start, end = month_bounds("2018-06")
    replay_scores = run_inference(final_df, data.feature_cols, start, end, model, verbose=False)
    replay_rows = final_df.set_index("order_id").loc[replay_scores["order_id"]].sort_values(["order_approved_dt", "order_id"])
    if history_ids.intersection(replay_rows.index):
        raise ValueError("June demonstration/replay orders overlap the refit population")
    if family == "linear":
        before = model.predict_proba(june_rows[data.feature_cols])
        rebind_expand_columns(model)
        if not np.array_equal(before, model.predict_proba(june_rows[data.feature_cols])):
            raise AssertionError("Rebinding the transformer changed predictions")

    # One deterministic randomly chosen order per approval date, never using labels.
    sample_rows = june_rows.groupby("order_approved_dt", group_keys=False).sample(n=1, random_state=42)
    if len(sample_rows) < 30:
        remainder = june_rows.drop(index=sample_rows.index).sample(n=30-len(sample_rows), random_state=42)
        sample_rows = pd.concat([sample_rows, remainder])
    sample_rows = sample_rows.sort_values(["order_approved_dt", "order_id"]).iloc[:30]
    samples = records(sample_rows[data.feature_cols].reset_index())
    numeric_ranges = {}
    for name in data.num_cols:
        values = data.history_df[name].dropna().astype(float)
        kind = "binary" if name.startswith("is_") else (
            "integer" if name.endswith("_count") or name.startswith("order_approved_")
            or name in ("payment_installments_max", "orders_approved_prev_7d", "promised_lead_days")
            else "continuous")
        numeric_ranges[name] = {
            "min": float(values.min()), "max": float(values.max()),
            "p01": float(values.quantile(.01)), "p99": float(values.quantile(.99)),
            "median": float(values.median()), "type": kind,
        }
    baseline = {name: details["median"] for name, details in numeric_ranges.items()}
    baseline.update({name: data.history_df[name].mode().iloc[0] for name in data.cat_cols})
    scores = june["predicted_probability"].to_numpy(dtype=float)
    counts, edges = np.histogram(scores, bins=30)
    score_distribution = {
        "n_orders": len(scores),
        "sorted_scores": sorted(scores.tolist()),
        "histogram": [{"lower": float(edges[i]), "upper": float(edges[i+1]), "count": int(count)}
                      for i, count in enumerate(counts)],
        "quantiles": {str(q): float(np.quantile(scores, q)) for q in (0, .01, .1, .25, .5, .75, .9, .95, .99, 1)},
        "top_10_percent_boundary": float(np.quantile(scores, .9)),
        "boundary_policy": "90th percentile of held-out June model scores only; no outcome-based threshold selection. Ties may yield more than 10% at or above the boundary.",
        "late_rate": float(june["actual_label"].mean()),
        "late_rate_note": "Historical context only; June outcomes never select samples, thresholds or model parameters.",
    }
    if family == "linear":
        global_importance = {"Coefficients": linear_importance(code)}
    else:
        global_importance = {split: records(pd.read_csv(OUTPUT / f"rf_permutation_importance_{split.lower()}.csv"))
                             for split in ("Validation", "June")}
    metrics = read_csv(OUTPUT / "model_results_long.csv")
    evaluation = metrics[metrics["model"].eq(code)].set_index("split").drop(columns="model").to_dict(orient="index")
    selection = read_csv(OUTPUT / "classification_model_selection_1.csv")
    confusion = read_csv(OUTPUT / "classification_confusion_1.csv")
    best_params = {"tree": run["tree_best_params"], "forest": run["rf_best_params"]}.get(family, {})
    metadata = {
        "model_code": code, "model_version": f"phase2-{code.lower()}-{FAMILY_NAMES[TASK][family]}-{RUN_DATE}",
        "family": family,
        "target": "late delivery according to the Phase 2 as-of label policy",
        "classes": {"0": "before_or_on_promise", "1": "late"},
        "positive_class": 1, "threshold": run["operating_threshold"], "threshold_policy": THRESHOLD_POLICY,
        "best_params": best_params, "random_state": RANDOM_STATE,
        "training_population": "Phase 2 known-outcome history before run date; canceled and unavailable orders excluded; June never fitted",
        "training_run_date": RUN_DATE, "train_rows": len(data.history_df),
        "training_order_ids_sha256": hashlib.sha256("\n".join(sorted(history_ids)).encode()).hexdigest(),
        "features": {"numeric": data.num_cols, "categorical": data.cat_cols},
        "selected_features": run["selected_variables"].get(code, data.feature_cols),
        "categories": {c: sorted(data.history_df[c].dropna().unique().tolist()) for c in data.cat_cols},
        "preprocessing": PREPROCESSING[family],
        "numeric_ranges": numeric_ranges,
        "baseline_features": baseline,
        "june_score_distribution": score_distribution,
        "global_importance": global_importance,
        "selection": records(selection), "selection_rule": run["selection_rule"],
        "selected_by_order_bootstrap": run["selected_by_order_bootstrap"],
        "confusion": records(confusion[confusion["Code"].eq(code)]),
        "false_negatives": records(read_csv(OUTPUT / "classification_false_negatives_1.csv")),
        "refit_check": records(read_csv(OUTPUT / "classification_refit_check_1.csv")),
        "fingerprints": {"artifact": actual_fingerprint, "notebook_reference": reference_fingerprint},
        "serialization": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
        "evaluation": evaluation,
        "evaluation_source": "results/simple_to_complex/classification/model_results_long.csv",
        "evaluation_notes": "Train/Validation use development fits; June uses history refit; July/August are monitoring. Metrics are historical, not live accuracy.",
        "export_verification": {"june_ap": reproduced["avg_precision"], "reference_june_ap": expected_ap,
                                "june_scored": len(june), "sample_training_overlap": 0},
    }
    save_bundle(model, metadata, output_dir / "classification_model.joblib", sample_rows[data.feature_cols],
                lambda m, x: m.predict_proba(x))
    (output_dir / "classification_model_stats.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples_dir = ROOT / "deployment/app/data"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "classification_sample_orders.json").write_text(json.dumps(samples, indent=2) + "\n")
    replay = {"model_version": metadata["model_version"], "days": {}}
    for day, rows in replay_rows.groupby("order_approved_dt", sort=True):
        replay["days"][day.strftime("%Y-%m-%d")] = records(rows[data.feature_cols].reset_index())
    labels = june.set_index("order_id")["actual_label"]
    outcomes = {"model_version": metadata["model_version"], "display_only": True,
                "usage": "Realised outcomes for historical display only; never used for fitting, tuning, sample selection or threshold selection. Canceled or unavailable orders are marked \"canceled\" and have no delivery outcome.",
                "outcomes": {order_id: "canceled" if status in EXCLUDED_STATUSES else (None if pd.isna(labels[order_id]) else int(labels[order_id]))
                             for order_id, status in replay_rows["order_status"].items()}}
    (samples_dir / "classification_june_orders.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    (samples_dir / "classification_june_outcomes.json").write_text(json.dumps(outcomes, separators=(",", ":")) + "\n")
    print(f"Saved {output_dir / 'classification_model.joblib'}; {code}, {len(data.history_df):,} refit rows; June AP {reproduced['avg_precision']:.12f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
