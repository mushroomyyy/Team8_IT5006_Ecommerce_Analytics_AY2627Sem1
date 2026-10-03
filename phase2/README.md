# Phase 2 — Modelling

Dual framing of one delivery-performance problem:
- **Classification** (`model_classification_dev.ipynb`): will an order arrive after its estimated delivery date?
- **Regression** (`model_regression_dev.ipynb`): how many days from approval to customer delivery, capped at 45 days?

## Layout
| Path | Contents |
|---|---|
| `model_classification_dev.ipynb` | Classification pipeline, label availability audit, pooled monthly backtest; time-series CV, hyperparameter tuning and threshold selection (saves `results/classification_tuning.json`) |
| `classification_evaluation.ipynb` | Classification model comparison and seller/product history check |
| `model_regression_dev.ipynb` | Lead-time target and horizon audit, 45-day chronological hold-out and time-series CV, hyperparameter tuning, champion selection by CV MAE, pooled monthly backtest that re-selects the champion at each run date (saves `results/regression_tuning.json`) |
| `src/data.py` | Locates and loads the Olist CSVs (extracts the bundled zip on first run) |
| `src/features.py` | `build_feature_table` (v2's `final_df`), as-of seller/product history, `add_extra_features`, feature lists |
| `src/labels.py` | `get_required_dates`, `labels_as_of`, `regression_targets_as_of`, cohort evaluation helpers |
| `src/splits.py` | `chronological_split`, `day_blocked_time_series_folds` |
| `src/evaluation.py` | Classification, regression and decile coverage metrics |
| `src/report_figures.py` | Report figures for the regression notebook; each call draws inline and saves a PNG to `results/figures/` |
| `src/tuning.py` | Default classifier pipelines, search spaces, CV summary, random search, out-of-fold threshold selection |

`src/` holds the v2 helpers without behaviour changes: at run date 2018-06-02 it reproduces v2's 39,960 window orders and 39,956 known labels.

`classification_evaluation.ipynb` uses the 31 base predictors plus four seller/product history predictors for every classifier: mean smoothed late rate and no-history share for each entity type. Its feature check compares this set with the original 31 predictors and with each history group separately on the same development months. History is reconstructed for each order's approval day; validation orders are excluded from fitted histories, and scored-month orders are excluded from that month's histories. The number of known prior orders is used to smooth each late rate but is not a model predictor. Entities with no known earlier outcomes receive the overall late rate known that day and an explicit no-history share. Orders without item records receive the same fallback, so the no-history share also includes missing entity identities.

## Running
```bash
pip install -r requirements.txt matplotlib seaborn jupyter
cd phase2
jupyter nbconvert --to notebook --execute model_regression_dev.ipynb   # or open it in Jupyter
python -m unittest test_tuning test_classification_eval test_historical_features
```
Notebooks add `phase2/` to `sys.path`, so `from src... import ...` works from either the project root or `phase2/`.

## Leakage rules
- Features must be known at approval time. `LEAKAGE_COLS` in `src/features.py` lists the outcome columns, and the notebooks assert none of them are used.
- Targets are assigned only from dates before the run date (`labels_as_of`, `regression_targets_as_of`).
- The test set is always the newest approval days. CV folds validate on days after their training days.
