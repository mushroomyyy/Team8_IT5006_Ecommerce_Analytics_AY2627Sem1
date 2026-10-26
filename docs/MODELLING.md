# Phase 2: Modelling late deliveries, simple to complex

Two views of one delivery-performance problem on the Olist e-commerce data, both predicted at order approval:
- **Classification:** will the order arrive after its estimated delivery date?
- **Regression:** by how many days will it arrive before or after that date (days from the promise)?

Each task starts from a baseline, builds a plain linear model (with transformations and variable selection judged on later time periods), then a single decision tree, and only then asks whether a Random Forest earns its extra complexity. Primary metrics: **AP** (area under the precision-recall curve) for classification and **RMSE** in days for regression.

Canceled and unavailable orders never receive a delivery outcome, so both tasks exclude them from training and from every evaluation set. The deployed app still scores every approved order, because a cancellation is not known at approval.

## Protocol
Every model is scored in the same way on the same orders. Tuning and variable selection (selected variables, decision-tree and Random Forest hyperparameters) use Train and its CV folds only; the choice between models uses Validation only. Linear-model transformations (log1p on skewed inputs, sine/cosine for month and weekday, no squared terms) are pre-specified from the predictors' distributions, not tuned on the outcome; variable selection is the only outcome-driven step.

| Name | Meaning |
|---|---|
| Train | Approvals from 18 Apr 2017 to 1 Feb 2018 whose outcomes are known (in-sample scores) |
| CV | 5 expanding, chronological folds inside Train: tuning and variable selection within each model family |
| Validation | 30-day out-of-time holdout, 19 Mar to 17 Apr 2018: the only set that chooses between models |
| Test (June) | June 2018 orders, scored by the chosen specification refitted on 2 June on all orders approved 18 Apr 2017 to 17 Apr 2018 (Train, gap and Validation; labels from deliveries recorded up to 1 June) |
| Monitoring | July and August 2018, scored by the frozen June models (no refit, retune or reselection) |

**Model choice (fixed before any result was seen).** The leader is the best Validation score. The 30 Validation approval days are resampled with replacement 2,000 times, every model is rescored on the same resamples, and each model's difference from the leader gets a 95% percentile interval. A model whose interval includes 0 is tied with the leader; among tied models the simplest wins (linear, then single tree, then forest; within a family, fewer fitted parameters). An order-level bootstrap is reported as a sensitivity check. CV scores of tuned or selected models are optimistic because those choices were made on the same folds, so CV never chooses between families. `src/model_selection.py` implements the rule; the notebooks export the decision table and record `selected_model` in `run_metadata.json`.

The classification operating cut-off is the F1-maximising probability on Validation for the selected model, applied unchanged to June.

## Models
| Classification | Regression |
|---|---|
| C0 no-skill (prior) | R0a median baseline; R0b Olist promise |
| C1 logistic, all 24 features | R1 OLS, all 24 features |
| C2 logistic, CV-stepwise subset (1-SE rule) | R2 OLS, CV-stepwise subset (1-SE rule) |
| C2B logistic, BIC-stepwise subset (supporting) | R2B OLS, BIC-stepwise subset (supporting) |
| C4 single decision tree, tuned (AP) | R4 single decision tree, tuned (RMSE) |
| C3 Random Forest, tuned (AP) | R3 Random Forest, tuned (RMSE) |
| C3d Random Forest, scikit-learn defaults | R3d Random Forest, scikit-learn defaults |

The model key (code, short label, description, feature count) is the first section of `results/simple_to_complex/RESULTS_SUMMARY.md`; the labels live in `src/model_labels.py`, and every table and figure shows "Code · short label".

Lasso appears only in the variable-selection overlap tables. Linear models always use log1p, cyclic calendar terms and no squared terms, and fold the 118 `Mixed` route orders into `All interstate` (no late orders in Train).

## Layout
| Path | Contents |
|---|---|
| `notebooks/01_data_features_multicollinearity.ipynb` | Data, features, label audits, multicollinearity, linear transforms, split and CV diagrams |
| `notebooks/02_classification.ipynb` | C0 to C4, class-imbalance experiment, model choice, deciles, confusion matrices and missed late orders, ROC/PR, drift, interpretation |
| `notebooks/03_regression.ipynb` | R0a to R4, model choice, dual-framing late flag, OLS diagnostics, drift, interpretation |
| `notebooks/04_summary.ipynb` | Reads the exports only: ladder figures and `RESULTS_SUMMARY.md` |
| `src/data.py` | Locates and loads the Olist CSVs |
| `src/features.py` | Order-level feature table, extra features and `LEAKAGE_COLS` |
| `src/labels.py` | Run-date windows and as-of-run-date targets |
| `src/splits.py` | Time-aware splits and CV folds |
| `src/datasets.py` | One shared definition of Train, Validation, CV folds and history per task |
| `src/preprocessing.py` | Preprocessing shared by both tasks |
| `src/linear_transforms.py` | Log and cyclic transforms for the linear models |
| `src/feature_selection.py` | Fixed multicollinearity screen (also a script) |
| `src/variable_selection.py` | CV-stepwise (1-SE), IC-stepwise, Lasso and overlap, Train only |
| `src/tuning.py` | Plain unweighted logistic regression with a convergence check |
| `src/regression_models.py` | Olist promise baseline, waiting-period clipping, pipeline builders |
| `src/rf_search.py` | Two-stage Random Forest search with cached outputs (also a script) |
| `src/tree_search.py` | Decision-tree grid search with cached outputs (also a script) |
| `src/model_selection.py` | Validation bootstrap model choice and fold-wise paired differences (Nadeau-Bengio corrected) |
| `src/resampling.py` | Resampling inside CV training folds only |
| `src/inference.py` | Daily inference over a date range |
| `src/evaluation.py` | Metrics, comparison, gap, drift and success-criteria tables, operating cut-off, confusion counts, false-negative breakdown, fingerprints |
| `src/interpretation.py` | statsmodels coefficient tables, hypothesis check, permutation importance, PDPs |
| `src/data_audits.py` | Data, label and transform audits for notebook 01 |
| `src/validation_timelines.py` | Train / CV / Validation / Test / Monitoring timeline figure |
| `src/report_figures.py`, `src/model_figures.py` | Figure style and the shared model figures |
| `src/model_labels.py` | Short labels, descriptions and the model key for every model code |
| `src/report_tables.py` | Export tables as HTML, LaTeX and CSV |
| `src/summary.py` | Ladder figures, `RESULTS_SUMMARY.md` and merged metadata (also a script) |
| `tests/phase2/test_*.py` | Unit tests (`python -m unittest`) |
| `results/feature_selection/` | Multicollinearity screen tables |
| `results/simple_to_complex/data/` | Notebook 01 tables |
| `results/simple_to_complex/classification/`, `regression/` | Tables (HTML, TeX, CSV), decision-tree and RF search caches, `run_metadata.json` per task |
| `results/simple_to_complex/figures/` | PNG figures at 200 dpi, prefixed `01_` to `04_` by notebook |
| `results/simple_to_complex/RESULTS_SUMMARY.md` | Generated summary of everything the report needs |
| `results/simple_to_complex/run_metadata.json` | Both tasks' metadata and package versions |
| `docs/MULTICOLLINEARITY_NOTES.md` | Notes on the feature funnel and multicollinearity decisions |
| `archive/phase2_archive/` | Earlier work, kept for restoring (below) |

### Archive
- `archive/phase2_archive/2026-10-08_before_feature_selection/`: the earlier notebooks (`model_*_dev.ipynb`, `model_validation_timelines.ipynb`), `src/`, `requirements.txt` and `README.md` as of `origin/main` before feature selection.
- `archive/phase2_archive/2026-10-09_before_simple_to_complex/`: the notebooks, `src/`, tests, old `results/` (figures, report tables, tuning JSON) and untracked prediction CSVs (`local_untracked/`) before this rebuild, with `snapshot_manifest.json`.

To inspect an earlier version, use its snapshot under `archive/phase2_archive/` or the historical commit recorded in its `snapshot_manifest.json`. Archived sources retain their original paths and are not current runnable notebooks.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-modelling.txt
pip install pytest
```
Data: `src/data.py` reads `Olist_CSV/` at the repository root, or extracts `data/olist_csv.zip` there on first use.

## Running
From the repository root, open the notebooks in Jupyter and run 01, 02, 03, then 04. Their first setup cell also supports a kernel launched from `notebooks/`. Saved outputs are submission evidence; full execution can overwrite result exports and is not needed to use the deployed models.

```bash
jupyter lab
python -m pytest tests/phase2 -q
```

Optional regeneration commands (write results; not part of a path-only migration):

```bash
python -m src.feature_selection
python -m src.rf_search --task classification
python -m src.tree_search --task classification
python -m src.summary
```

Notebooks 02 and 03 load the cached Random Forest and decision-tree searches (`rf_best_params.json` and `tree_best_params.json` in each task folder) with `RUN_RF_SEARCH = False` and `RUN_TREE_SEARCH = False`; run `src.rf_search` or `src.tree_search` only to rebuild them.

Expected runtimes on a laptop: notebook 01 a few minutes; notebooks 02 and 03 about 5 to 25 minutes each; `src.rf_search` about 5 to 10 minutes per task; `src.tree_search` about 1 minute per task; notebook 04 and `src.summary` seconds; `python -m unittest` about 5 seconds.

## Leakage rules
- Features must be known at approval time. `LEAKAGE_COLS` in `src/features.py` lists the outcome columns, and the notebooks assert none is used.
- Targets are assigned only from dates before the run date (`labels.py`).
- Selection, transforms and tuning see Train rows only. Validation rows enter only the choice between models and the operating cut-off; Test (June) rows never enter any choice, and Monitoring months are scored by frozen models (fingerprints are checked unchanged).
- Canceled and unavailable orders are excluded from training and evaluation in both tasks; they have no delivery outcome.
- Item, product, seller and payment tables are final snapshots rather than their state at approval; this is a dataset limitation.
- All models are scored on identical Validation and June order IDs.
