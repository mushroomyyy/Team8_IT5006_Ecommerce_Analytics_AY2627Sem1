# Phase 2: Modelling late deliveries, simple to complex

Two views of one delivery-performance problem on the Olist e-commerce data, both predicted at order approval:
- **Classification:** will the order arrive after its estimated delivery date?
- **Regression:** by how many days will it arrive before or after that date (days from the promise)?

Each task starts from a baseline, builds a plain linear model (with transformations and variable selection judged on later time periods), and only then asks whether a Random Forest earns its extra complexity. Primary metrics: **AP** (area under the precision-recall curve) for classification and **RMSE** in days for regression.

## Protocol
Every model is scored in the same way on the same orders. Every decision (transforms, selected variables, Random Forest hyperparameters) uses Train only.

| Name | Meaning |
|---|---|
| Train | Approvals from 18 Apr 2017 to 1 Feb 2018 whose outcomes are known (in-sample scores) |
| CV | 5 expanding, chronological folds inside Train (used for all choices) |
| Validation | 30-day out-of-time holdout, 19 Mar to 17 Apr 2018 (diagnostic only) |
| Test (June) | June 2018 orders, scored by models refitted on all known history before 2 June |
| Monitoring | July and August 2018, scored by the frozen June models (no refit, retune or reselection) |

**No model is selected or called a winner.** The results show how performance changes from simple to complex models.

## Models
| Classification | Regression |
|---|---|
| C0 no-skill (prior) | R0a median baseline; R0b Olist promise |
| C1 logistic, all 24 features | R1 OLS, all 24 features |
| C2 logistic, CV-stepwise subset (1-SE rule) | R2 OLS, CV-stepwise subset (1-SE rule) |
| C2B logistic, BIC-stepwise subset (supporting) | R2B OLS, BIC-stepwise subset (supporting) |
| C3 Random Forest, tuned (AP) | R3 Random Forest, tuned (RMSE) |
| C3d Random Forest, scikit-learn defaults | R3d Random Forest, scikit-learn defaults |

Lasso appears only in the variable-selection overlap tables. Linear models use log, cyclic calendar and squared terms where CV supports them, and fold the 118 `Mixed` route orders into `All interstate` (no late orders in Train).

## Layout
| Path | Contents |
|---|---|
| `01_data_features_multicollinearity.ipynb` | Data, features, label audits, multicollinearity, linear transforms, split and CV diagrams |
| `02_classification.ipynb` | C0 to C3d, class-imbalance experiment, deciles, ROC/PR, drift, interpretation |
| `03_regression.ipynb` | R0a to R3d, dual-framing late flag, OLS diagnostics, drift, interpretation |
| `04_summary.ipynb` | Reads the exports only: ladder figures and `RESULTS_SUMMARY.md` |
| `src/data.py` | Locates and loads the Olist CSVs |
| `src/features.py` | Order-level feature table, extra features and `LEAKAGE_COLS` |
| `src/labels.py` | Run-date windows and as-of-run-date targets |
| `src/splits.py` | Time-aware splits and CV folds |
| `src/datasets.py` | One shared definition of Train, Validation, CV folds and history per task |
| `src/preprocessing.py` | Preprocessing shared by both tasks |
| `src/linear_transforms.py` | Log, cyclic and squared transforms for the linear models |
| `src/feature_selection.py` | Fixed multicollinearity screen (also a script) |
| `src/variable_selection.py` | CV-stepwise (1-SE), IC-stepwise, Lasso and overlap, Train only |
| `src/tuning.py` | Plain unweighted logistic regression with a convergence check |
| `src/regression_models.py` | Olist promise baseline, waiting-period clipping, pipeline builders |
| `src/rf_search.py` | Two-stage Random Forest search with cached outputs (also a script) |
| `src/resampling.py` | Resampling inside CV training folds only |
| `src/inference.py` | Daily inference over a date range |
| `src/evaluation.py` | Metrics, comparison, gap, drift and success-criteria tables, fingerprints |
| `src/interpretation.py` | statsmodels coefficient tables, hypothesis check, permutation importance, PDPs |
| `src/data_audits.py` | Data, label and transform audits for notebook 01 |
| `src/validation_timelines.py` | Train / CV / Validation / Test / Monitoring timeline figure |
| `src/report_figures.py`, `src/model_figures.py` | Figure style and the shared model figures |
| `src/report_tables.py` | Export tables as HTML, LaTeX and CSV |
| `src/summary.py` | Ladder figures, `RESULTS_SUMMARY.md` and merged metadata (also a script) |
| `test_*.py` | Unit tests (`python -m unittest`) |
| `results/feature_selection/` | Multicollinearity screen tables |
| `results/simple_to_complex/data/` | Notebook 01 tables |
| `results/simple_to_complex/classification/`, `regression/` | Tables (HTML, TeX, CSV), RF search cache, `run_metadata.json` per task |
| `results/simple_to_complex/figures/` | PNG figures at 200 dpi, prefixed `01_` to `04_` by notebook |
| `results/simple_to_complex/RESULTS_SUMMARY.md` | Generated summary of everything the report needs |
| `results/simple_to_complex/run_metadata.json` | Both tasks' metadata and package versions |
| `MULTICOLLINEARITY_NOTES.md` | Notes on the feature funnel and multicollinearity decisions |
| `Archive/` | Earlier work, kept for restoring (below) |

### Archive
- `Archive/2026-10-08_before_feature_selection/`: the earlier notebooks (`model_*_dev.ipynb`, `model_validation_timelines.ipynb`), `src/`, `requirements.txt` and `README.md` as of `origin/main` before feature selection.
- `Archive/2026-10-09_before_simple_to_complex/`: the notebooks, `src/`, tests, old `results/` (figures, report tables, tuning JSON) and untracked prediction CSVs (`local_untracked/`) before this rebuild, with `snapshot_manifest.json`.

To restore, copy the files you need back to `phase2/` (for example `cp -R Archive/2026-10-09_before_simple_to_complex/src/. src/`), or run `git checkout <commit> -- phase2/<path>` with the commit recorded in that folder's `snapshot_manifest.json`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data: `src/data.py` reads the Olist CSVs from `Olist_CSV/` one level above `phase2/`. If that folder is missing, it extracts `streamlit_release/data/olist_csv.zip` (also one level above `phase2/`) into it on first run.

## Running
From `phase2/`:
```bash
jupyter nbconvert --to notebook --execute --inplace 01_data_features_multicollinearity.ipynb --ExecutePreprocessor.timeout=-1
# repeat for 02_classification.ipynb, 03_regression.ipynb and 04_summary.ipynb, in that order
python -m src.feature_selection                 # multicollinearity screen -> results/feature_selection/
python -m src.rf_search --task classification   # cached RF search; same for --task regression
python -m src.summary                           # figures 04_*, RESULTS_SUMMARY.md, run_metadata.json
python -m unittest                              # all tests
```
Notebooks 02 and 03 load the cached Random Forest search (`rf_best_params.json` in each task folder) with `RUN_RF_SEARCH = False`; run `src.rf_search` only to rebuild it.

Expected runtimes on a laptop: notebook 01 a few minutes; notebooks 02 and 03 about 5 to 25 minutes each; `src.rf_search` about 10 minutes per task; notebook 04 and `src.summary` seconds; `python -m unittest` about 5 seconds.

## Leakage rules
- Features must be known at approval time. `LEAKAGE_COLS` in `src/features.py` lists the outcome columns, and the notebooks assert none is used.
- Targets are assigned only from dates before the run date (`labels.py`).
- Selection, transforms and tuning see Train rows only. Validation and Test (June) rows never enter them, and Monitoring months are scored by frozen models (fingerprints are checked unchanged).
- All models are scored on identical Validation and June order IDs.
