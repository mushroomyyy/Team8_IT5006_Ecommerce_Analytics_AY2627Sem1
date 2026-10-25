"""Export C3 from the repository root with -m deployment.export.train_classification.

Uses the notebook's cached CV-selected parameters and known-outcome history,
never June labels for fitting or threshold selection. Raw Olist data is needed
only for this offline export, not for the deployed API.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import RANDOM_STATE  # noqa: E402
from src.data import load_olist_data  # noqa: E402
from src.datasets import RUN_DATE, build_task_data  # noqa: E402
from src.evaluation import score_months, classification_eval_metrics, fingerprint  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402
from src.rf_search import tuned_rf_pipeline  # noqa: E402

RESULTS = ROOT / "results/simple_to_complex/classification"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
THRESHOLD_POLICY = (
    "Illustrative fixed 0.5 cutoff; not optimized for operational use. "
    "Top-10% evaluation ranks an entire cohort and is not this single-order cutoff."
)


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-ready records without pandas' default ten-digit float rounding."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def export(output_dir: Path = ARTIFACTS) -> dict:
    source = json.loads((RESULTS / "run_metadata.json").read_text())
    params = json.loads((RESULTS / "rf_best_params.json").read_text())["best_params"]
    if params != source["rf_best_params"]:
        raise ValueError("Cached C3 parameters disagree with the notebook run metadata")
    olist = load_olist_data(verbose=False)
    final_df = add_extra_features(build_feature_table(olist), olist)
    data = build_task_data("classification", final_df)
    if len(data.history_df) != source["row_counts"]["history"]:
        raise ValueError("Refit population no longer matches the Phase 2 run")
    model = tuned_rf_pipeline("classification", params, data.num_cols, data.cat_cols, n_jobs=1)
    model.fit(data.history_df[data.feature_cols], data.history_df["y"])

    # Score exactly the same June eligibility and as-of labels as notebook 02.
    june = score_months(model, final_df, olist["orders"], data.feature_cols, ["2018-06"])["2018-06"]
    reproduced = classification_eval_metrics(
        june["actual_label"], june["predicted_probability"], order_ids=june["order_id"])
    expected_ap = source["headline_june"]["C3"]["avg_precision"]
    if not np.isclose(reproduced["avg_precision"], expected_ap, atol=1e-10, rtol=0):
        raise ValueError(f"June AP {reproduced['avg_precision']} differs from notebook {expected_ap}")

    actual_fingerprint = fingerprint(model)
    notebook_model = copy.deepcopy(model).set_params(classifier__n_jobs=-1)
    normalized_fingerprint = fingerprint(notebook_model)
    reference_fingerprint = source["fingerprints"]["C3"]
    if normalized_fingerprint != reference_fingerprint:
        raise AssertionError("Exported C3 fitted-state fingerprint differs from the notebook")

    # round_trip parsing is exact; pandas' default fast parser differs across CPU platforms.
    metric_frame = pd.read_csv(RESULTS / "model_results_long.csv", float_precision="round_trip")
    evaluation = metric_frame[metric_frame["model"].eq("C3")].set_index("split").drop(columns="model")
    evaluation = evaluation.to_dict(orient="index")
    history_ids = set(data.history_df["order_id"])
    june_rows = final_df.set_index("order_id").loc[june["order_id"]].copy()
    june_rows = june_rows.sort_values(["order_approved_dt", "order_id"])
    # One deterministic randomly chosen order per approval date, never using labels.
    sample_rows = june_rows.groupby("order_approved_dt", group_keys=False).sample(n=1, random_state=42)
    if len(sample_rows) < 30:
        remainder = june_rows.drop(index=sample_rows.index).sample(n=30-len(sample_rows), random_state=42)
        sample_rows = pd.concat([sample_rows, remainder])
    sample_rows = sample_rows.sort_values(["order_approved_dt", "order_id"]).iloc[:30]
    if history_ids.intersection(june_rows.index):
        raise ValueError("June demonstration/replay orders overlap the refit population")
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
    global_importance = {}
    for split in ("Validation", "June"):
        path = RESULTS / f"rf_permutation_importance_{split.lower()}.csv"
        if path.exists():
            global_importance[split] = records(pd.read_csv(path))
    metadata = {
        "model_code": "C3", "model_version": "phase2-c3-rf-2018-06-02",
        "target": "late delivery according to the Phase 2 as-of label policy",
        "classes": {"0": "before_or_on_promise", "1": "late"},
        "positive_class": 1, "threshold": 0.5, "threshold_policy": THRESHOLD_POLICY,
        "best_params": params, "random_state": RANDOM_STATE,
        "training_population": "Phase 2 known-outcome history before run date; June never fitted",
        "training_run_date": RUN_DATE, "train_rows": len(data.history_df),
        "training_order_ids_sha256": hashlib.sha256("\n".join(sorted(history_ids)).encode()).hexdigest(),
        "features": {"numeric": data.num_cols, "categorical": data.cat_cols},
        "categories": {c: sorted(data.history_df[c].dropna().unique().tolist()) for c in data.cat_cols},
        "preprocessing": "Raw numeric median imputation and reference-coded one-hot encoding. API rejects null and unknown categorical values before encoding.",
        "numeric_ranges": numeric_ranges,
        "baseline_features": baseline,
        "june_score_distribution": score_distribution,
        "global_importance": global_importance,
        "fingerprints": {"artifact": actual_fingerprint, "notebook_normalized": normalized_fingerprint,
                         "notebook_reference": reference_fingerprint},
        "serialization": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "evaluation": evaluation,
        "evaluation_source": "results/simple_to_complex/classification/model_results_long.csv",
        "evaluation_notes": "Train/Validation use development fits; June uses history refit; July/August are monitoring. Metrics are historical, not live accuracy.",
        "export_verification": {"june_ap": reproduced["avg_precision"], "reference_june_ap": expected_ap,
                                "june_scored": len(june), "sample_training_overlap": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "classification_c3_rf.joblib"
    joblib.dump({"model": model, "metadata": metadata}, artifact, compress=3)
    reloaded = joblib.load(artifact)["model"]
    X_check = sample_rows[data.feature_cols]
    if not np.array_equal(model.predict_proba(X_check), reloaded.predict_proba(X_check)):
        raise AssertionError("Serialization changed predictions")
    (output_dir / "classification_model_stats.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples_dir = ROOT / "deployment/app/data"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "classification_sample_orders.json").write_text(json.dumps(samples, indent=2) + "\n")
    replay = {"model_version": metadata["model_version"], "days": {}}
    for day, rows in june_rows.groupby("order_approved_dt", sort=True):
        replay["days"][day.strftime("%Y-%m-%d")] = records(rows[data.feature_cols].reset_index())
    outcomes = {"model_version": metadata["model_version"], "display_only": True,
                "usage": "Realised outcomes for historical display only; never used for fitting, tuning, sample selection or threshold selection.",
                "outcomes": {str(row.order_id): (None if pd.isna(row.actual_label) else int(row.actual_label)) for row in june.itertuples()}}
    (samples_dir / "classification_june_orders.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    (samples_dir / "classification_june_outcomes.json").write_text(json.dumps(outcomes, separators=(",", ":")) + "\n")
    print(f"Saved {artifact}; {len(data.history_df):,} refit rows; June AP {reproduced['avg_precision']:.12f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
