# Phase 2 — Modelling

Dual framing of one delivery-performance problem:
- **Classification** (`model_classification_dev.ipynb`): will an order arrive after its estimated delivery date?
- **Regression** (`model_regression_dev.ipynb`): by how many days will an order arrive before or after its estimated delivery date? (Lead time capped at 45 days.)

## Layout
| Path | Contents |
|---|---|
| `model_classification_dev.ipynb` | Classification pipeline, label availability audit, pooled monthly backtest; time-series CV, hyperparameter tuning and threshold diagnostics, and a full-history default-versus-tuned June comparison |
| `model_regression_dev.ipynb` | Days-from-promise target and horizon audit, 30-day chronological hold-out and time-series CV, hyperparameter tuning, champion selection by hold-out MAE, pooled monthly backtest that re-selects the champion at each run date (saves `results/regression_tuning.json`) |
| `model_validation_timelines.ipynb` | Rebuilds the shared June validation timeline from actual eligible orders and saves `results/figures/09_validation_timelines.png` |
| `src/data.py` | Locates and loads the Olist CSVs (extracts the bundled zip on first run) |
| `src/features.py` | `build_feature_table` (v2's `final_df`), as-of seller/product history, `add_extra_features`, feature lists |
| `src/labels.py` | `get_required_dates`, `labels_as_of`, `regression_targets_as_of`, cohort evaluation helpers |
| `src/preprocessing.py` | `make_preprocessor`: median imputation, optional scaling and one-hot encoding, shared by both tracks |
| `src/splits.py` | `chronological_split`, `day_blocked_time_series_folds` |
| `src/evaluation.py` | Classification, regression and decile coverage metrics |
| `src/report_figures.py` | Report figures for the regression notebook; each call draws inline and saves a PNG to `results/figures/` |
| `src/tuning.py` | Default classifier pipelines, search spaces, CV summary, random search, out-of-fold threshold selection |

Both tracks use a 365-day historical window, a 30-day development holdout preceded by a 45-day gap, and five expanding CV folds with 30-calendar-day validation blocks and 45-day gaps. Training outcomes must be known at each validation boundary. The final refit uses the full eligible historical window. Default and tuned candidates use identical rows within each comparison.

## Running
```bash
pip install -r requirements.txt matplotlib seaborn jupyter
cd phase2
jupyter nbconvert --to notebook --execute model_regression_dev.ipynb   # or open it in Jupyter
python -m unittest test_tuning
```
Notebooks add `phase2/` to `sys.path`, so `from src... import ...` works from either the project root or `phase2/`.

## Leakage rules
- Features must be known at approval time. `LEAKAGE_COLS` in `src/features.py` lists the outcome columns, and the notebooks assert none of them are used.
- Targets are assigned only from dates before the run date (`labels_as_of`, `regression_targets_as_of`).
- The test set is always the newest approval days. CV folds validate on days after their training days.
