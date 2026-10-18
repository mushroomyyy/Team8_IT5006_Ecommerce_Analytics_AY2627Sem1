"""Fit and package the Phase 2 selected regression model (R1: all-feature OLS).

The fit uses the same June refit population as notebook 03: all known-outcome
history before 2 June 2018. The preprocessing is stored inside the joblib bundle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))

from src import RANDOM_STATE  # noqa: E402
from src.data import load_olist_data  # noqa: E402
from src.datasets import build_task_data  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402
from src.linear_transforms import make_linear_preprocessor, skewness_table  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(exist_ok=True)
    olist = load_olist_data(verbose=False)
    final_df = add_extra_features(build_feature_table(olist), olist)
    data = build_task_data("regression", final_df)
    skew_candidates = ["payment_value_sum", "order_frieght_value_sum", "order_product_weight_g_sum",
                       "order_product_volume_cm3_sum", "approval_lag_hours", "seller_dist_km_mean",
                       "freight_to_price_ratio", "orders_approved_prev_7d"]
    log_cols = skewness_table(data.train_df, skew_candidates).loc[
        lambda frame: frame["log1p_recommended"], "feature"].tolist()
    model = Pipeline([
        ("preprocessor", make_linear_preprocessor(
            data.num_cols, data.cat_cols, log_cols=log_cols,
            cyclic=True, merge_mixed_route=True,
        )),
        ("model", LinearRegression()),
    ])
    model.fit(data.history_df[data.feature_cols], data.history_df["y"])

    metadata = {
        "model_code": "R1",
        "model_version": "phase2-r1-ols-2026-10-09",
        "target": "days from the promised delivery date",
        "training_population": "Phase 2 history with known outcomes before 2018-06-02",
        "train_rows": int(len(data.history_df)),
        "random_state": RANDOM_STATE,
        "serialization": {"scikit_learn": sklearn.__version__, "python": sys.version.split()[0]},
        "features": {"numeric": data.num_cols, "categorical": data.cat_cols},
        "linear_transforms": {"log_cols": log_cols, "cyclic": True, "merge_mixed_route": True},
        "evaluation": {
            "validation_rmse": 8.205,
            "june_test_rmse": 7.369,
            "june_test_mae": 5.81,
            "june_reference_rf_rmse": 7.953,
            "source": "phase2/results/simple_to_complex/regression/regression_model_comparison_5.csv",
        },
    }
    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, output_dir / "regression_r1_ols.joblib", compress=3)
    (output_dir / "regression_model_stats.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {output_dir / 'regression_r1_ols.joblib'}")
    print(f"Packaged {len(data.history_df):,} rows and {len(data.feature_cols)} features")


if __name__ == "__main__":
    main()
