"""Export the selected regression model from the repository root with -m deployment.export.train_regression.

The model is the one notebook 03 chose on Validation (`selected_model` in run_metadata.json),
refitted on all known-outcome history before 2 June 2018. Raw Olist data is needed only
for this offline export, not for the deployed API.
"""
from __future__ import annotations

import argparse
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
from src.datasets import RUN_DATE, build_task_data  # noqa: E402
from src.evaluation import score_regression_months, score_regression_split  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402

TASK = "regression"
OUTPUT = RESULTS / TASK
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def export(output_dir: Path = ARTIFACTS) -> dict:
    run, code, family = selected_model(TASK)
    olist = load_olist_data(verbose=False)
    final_df = add_extra_features(build_feature_table(olist), olist)
    data = build_task_data(TASK, final_df)
    if len(data.history_df) != run["row_counts"]["history"]:
        raise ValueError("Refit population no longer matches the Phase 2 run")
    model = fit_selected(TASK, build_spec(TASK, run, code, family, data), family, data)

    # Same June eligibility, clipping and targets as notebook 03; fingerprint before any rebinding.
    june = score_regression_months(model, final_df, olist["orders"], data.feature_cols, ["2018-06"])["2018-06"]
    reproduced = score_regression_split(code, "June", june["actual_target"], june["predicted_days"], june["actual_label"])
    expected_rmse = run["headline_june"][code]["rmse"]
    if not np.isclose(reproduced["rmse"], expected_rmse, atol=1e-10, rtol=0):
        raise ValueError(f"June RMSE {reproduced['rmse']} differs from notebook {expected_rmse}")
    fitted_fingerprint = notebook_fingerprint(TASK, model, family)
    reference_fingerprint = run["fingerprints"][code]
    if fitted_fingerprint != reference_fingerprint:
        raise AssertionError(f"Exported {code} fitted-state fingerprint differs from the notebook")

    # Stable inference-only module path; every fitted value is preserved.
    june_rows = final_df.set_index("order_id").loc[june["order_id"], data.feature_cols]
    if family == "linear":
        before = model.predict(june_rows)
        rebind_expand_columns(model)
        if not np.array_equal(before, model.predict(june_rows)):
            raise AssertionError("Rebinding the transformer changed predictions")

    metrics = read_csv(OUTPUT / "model_results_long.csv")
    evaluation = metrics[metrics["model"].eq(code)].set_index("split").drop(columns="model").to_dict(orient="index")
    selection = read_csv(OUTPUT / "regression_model_selection_1.csv")
    refit_check = read_csv(OUTPUT / "regression_refit_check_1.csv")
    metadata = {
        "model_code": code,
        "model_version": f"phase2-{code.lower()}-{FAMILY_NAMES[TASK][family]}-{RUN_DATE}",
        "family": family,
        "target": "days from the promised delivery date; negative means early, positive means late",
        "training_population": "Phase 2 history with known outcomes before 2018-06-02; canceled and unavailable orders excluded",
        "training_run_date": RUN_DATE, "train_rows": int(len(data.history_df)), "random_state": RANDOM_STATE,
        "features": {"numeric": data.num_cols, "categorical": data.cat_cols},
        "selected_features": run["selected_variables"].get(code, data.feature_cols),
        "linear_transforms": run["linear_transforms"] if family == "linear" else None,
        "selection": records(selection), "selection_rule": run["selection_rule"],
        "selected_by_order_bootstrap": run["selected_by_order_bootstrap"],
        "refit_check": records(refit_check),
        "fingerprints": {"artifact": fitted_fingerprint, "notebook_reference": reference_fingerprint},
        "serialization": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
        "evaluation": evaluation,
        "evaluation_source": "results/simple_to_complex/regression/model_results_long.csv",
        "export_verification": {"june_rmse": reproduced["rmse"], "reference_june_rmse": expected_rmse,
                                "june_scored": len(june)},
    }
    save_bundle(model, metadata, output_dir / "regression_model.joblib", june_rows.head(500), lambda m, x: m.predict(x))
    (output_dir / "regression_model_stats.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output_dir / 'regression_model.joblib'}; {code}, {len(data.history_df):,} refit rows; June RMSE {reproduced['rmse']:.12f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
