"""Export C3 from the repository root with -m deployment.train_classification.

Uses the notebook's cached CV-selected parameters and known-outcome history,
never June labels for fitting or threshold selection. Raw Olist data is needed
only for this offline export, not for the deployed API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))

from src import RANDOM_STATE  # noqa: E402
from src.data import load_olist_data  # noqa: E402
from src.datasets import RUN_DATE, build_task_data  # noqa: E402
from src.evaluation import score_months, classification_eval_metrics  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402
from src.rf_search import tuned_rf_pipeline  # noqa: E402

RESULTS = ROOT / "phase2/results/simple_to_complex/classification"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD_POLICY = (
    "Illustrative fixed 0.5 cutoff; not optimized for operational use. "
    "Top-10% evaluation ranks an entire cohort and is not this single-order cutoff."
)


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

    metric_frame = pd.read_csv(RESULTS / "model_results_long.csv")
    evaluation = metric_frame[metric_frame["model"].eq("C3")].set_index("split").drop(columns="model")
    evaluation = json.loads(evaluation.to_json(orient="index"))
    history_ids = set(data.history_df["order_id"])
    sample_rows = final_df.set_index("order_id").loc[june["order_id"].head(30)]
    if history_ids.intersection(sample_rows.index):
        raise ValueError("Demonstration orders overlap the refit population")
    samples = json.loads(sample_rows[data.feature_cols].reset_index().to_json(orient="records"))
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
        "preprocessing": "Raw numeric median imputation and reference-coded one-hot encoding; unseen categories ignored",
        "serialization": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "evaluation": evaluation,
        "evaluation_source": "phase2/results/simple_to_complex/classification/model_results_long.csv",
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
    (output_dir / "classification_sample_orders.json").write_text(json.dumps(samples, indent=2) + "\n")
    print(f"Saved {artifact}; {len(data.history_df):,} refit rows; June AP {reproduced['avg_precision']:.12f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
