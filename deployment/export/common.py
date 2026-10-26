"""Shared steps of the two model exporters: rebuild the selected specification and fit it.

Both exporters read the Validation choice from the notebook's `run_metadata.json`
(`selected_model`) and refit it on the known-outcome history, exactly as the notebooks do.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import fingerprint  # noqa: E402
from src.rf_search import STEP, tuned_rf_pipeline  # noqa: E402
from src.tree_search import tuned_tree_pipeline  # noqa: E402
from src.tuning import fit_logistic_checked  # noqa: E402
from src.variable_selection import make_unit_pipeline_builder  # noqa: E402

RESULTS = ROOT / "results/simple_to_complex"
# Short labels mirror src/model_labels.py; the app and API never import src.
FAMILY_NAMES = {"classification": {"linear": "logistic", "tree": "tree", "forest": "rf"},
                "regression": {"linear": "ols", "tree": "tree", "forest": "rf"}}
DEPLOYABLE = {"classification": ("C1", "C2", "C2B", "C4", "C3"), "regression": ("R1", "R2", "R2B", "R4", "R3")}


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-ready records without pandas' default ten-digit float rounding."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def read_csv(path: Path) -> pd.DataFrame:
    # round_trip parsing is exact; pandas' default fast parser differs across CPU platforms.
    return pd.read_csv(path, float_precision="round_trip")


def selected_model(task: str):
    """(run metadata, code, family) of the model the notebook chose on Validation."""
    run = json.loads((RESULTS / task / "run_metadata.json").read_text())
    code = run["selected_model"]
    if code not in DEPLOYABLE[task]:
        raise ValueError(f"{code} cannot be deployed; expected one of {DEPLOYABLE[task]}")
    selection = read_csv(RESULTS / task / f"{task}_model_selection_1.csv")
    family = selection.set_index("Code").loc[code, "family"]
    return run, code, family


def build_spec(task: str, run: dict, code: str, family: str, data):
    """Unfitted pipeline for the selected model, as built in the notebooks."""
    if family == "linear":
        build = make_unit_pipeline_builder(task, data.num_cols, data.cat_cols,
                                           log_cols=run["linear_transforms"]["log_cols"],
                                           cyclic=True, merge_mixed_route=True)
        return build(run["selected_variables"][code])
    if family == "tree":
        return tuned_tree_pipeline(task, run["tree_best_params"], data.num_cols, data.cat_cols)
    return tuned_rf_pipeline(task, run["rf_best_params"], data.num_cols, data.cat_cols, n_jobs=1)


def fit_selected(task: str, spec, family: str, data):
    """Refit on all known-outcome history before the run date (the June refit)."""
    X, y = data.history_df[data.feature_cols], data.history_df["y"]
    if task == "classification" and family == "linear":
        return fit_logistic_checked(spec, X, y)[0]
    return clone(spec).fit(X, y)


def notebook_fingerprint(task: str, model, family: str) -> str:
    """Fingerprint as the notebook computed it (forests there used n_jobs=-1)."""
    if family == "forest":
        model = copy.deepcopy(model).set_params(**{f"{STEP[task]}__n_jobs": -1})
    return fingerprint(model)


def rebind_expand_columns(model) -> None:
    """Point the fitted linear transformer at the standalone API module; no refitting."""
    from deployment.api.transforms import ExpandColumns
    model.named_steps["preprocessor"].named_steps["expand"].__class__ = ExpandColumns


def save_bundle(model, metadata: dict, artifact: Path, check_frame: pd.DataFrame, predict) -> None:
    """Write the joblib bundle and require the reloaded model to predict identically."""
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact, compress=3)
    if not np.array_equal(predict(model, check_frame), predict(joblib.load(artifact)["model"], check_frame)):
        raise AssertionError("Serialization changed predictions")
