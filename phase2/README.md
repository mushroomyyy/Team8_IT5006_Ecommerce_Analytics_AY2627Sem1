# Phase 2 — Modelling

Dual framing of one delivery-performance problem:
- **Classification** (`model_dev_v2.ipynb`): will an order arrive after its estimated delivery date?
- **Regression** (`regression_dev.ipynb`): how many days from approval to customer delivery, capped at 45 days?

## Layout
| Path | Contents |
|---|---|
| `model_dev_v2.ipynb` | Classification pipeline, label availability audit, monthly inference |
| `regression_dev.ipynb` | Lead-time target and horizon audit, chronological split and CV, tuning, backtest |
| `src/data.py` | Locates and loads the Olist CSVs (extracts the bundled zip on first run) |
| `src/features.py` | `build_feature_table` (v2's `final_df`), `add_extra_features`, feature lists |
| `src/labels.py` | `get_required_dates`, `labels_as_of`, `regression_targets_as_of`, cohort evaluation helpers |
| `src/splits.py` | `chronological_split`, `day_blocked_time_series_folds` |
| `src/evaluation.py` | Classification, regression and decile coverage metrics |

`src/` holds the v2 helpers without behaviour changes: at run date 2018-06-02 it reproduces v2's 39,960 window orders and 39,956 known labels.

## Running
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn jupyter
cd phase2
jupyter nbconvert --to notebook --execute regression_dev.ipynb   # or open it in Jupyter
```
Notebooks add `phase2/` to `sys.path`, so `from src... import ...` works from either the project root or `phase2/`.

## Leakage rules
- Features must be known at approval time. `LEAKAGE_COLS` in `src/features.py` lists the outcome columns, and the notebooks assert none of them are used.
- Targets are assigned only from dates before the run date (`labels_as_of`, `regression_targets_as_of`).
- The test set is always the newest approval days. CV folds validate on days after their training days.
