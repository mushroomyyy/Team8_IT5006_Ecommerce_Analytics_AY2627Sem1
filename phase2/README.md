# Phase 2 — Modelling

Dual framing of one delivery-performance problem:
- **Classification** (`model_classification_dev.ipynb`): will an order arrive after its estimated delivery date?
- **Regression** (`model_regression_dev.ipynb`): by how many days will an order arrive before or after its estimated delivery date? (Lead time capped at 45 days.)

## Layout
| Path | Contents |
|---|---|
| `model_classification_dev.ipynb` | Classification development, August model selection and evaluation, and optional June–August comparison |
| `model_regression_dev.ipynb` | Days-from-promise development, August model selection and evaluation, and optional sensitivity and June–August comparisons |
| `model_validation_timelines.ipynb` | Rebuilds the shared August validation timeline from actual eligible orders and saves `results/figures/09_validation_timelines.png` |
| `src/data.py` | Locates and loads the Olist CSVs (extracts the bundled zip on first run) |
| `src/features.py` | `build_feature_table` (v2's `final_df`), as-of seller/product history, `add_extra_features`, feature lists |
| `src/labels.py` | `get_required_dates`, `labels_as_of`, `regression_targets_as_of`, cohort evaluation helpers |
| `src/preprocessing.py` | `make_preprocessor`: median imputation, optional scaling and one-hot encoding, shared by both tracks |
| `src/splits.py` | `chronological_split`, `day_blocked_time_series_folds` |
| `src/evaluation.py` | Classification, regression and decile coverage metrics |
| `src/report_tables.py` | Shared export of formatted tables to Word-friendly HTML, booktabs LaTeX and CSV |
| `results/report_tables/` | August report tables and summary CSVs; optional monthly comparisons use a separate subdirectory, and order-level predictions stay local |
| `src/report_figures.py` | Report figures for the regression notebook; each call draws inline and saves a PNG to `results/figures/` |
| `src/tuning.py` | Default classifier pipelines, search spaces, CV summary, random search, out-of-fold threshold selection |

Both tracks use a 365-day historical window, a 30-day development holdout preceded by a 45-day gap, and five expanding CV folds with 30-calendar-day validation blocks and 45-day gaps. Training outcomes must be known at each validation boundary. The final refit uses the full eligible historical window. Default and tuned candidates use identical rows within each comparison.

## August evaluation and report tables

The main workflow in each modelling notebook uses a run date of **2 August 2018**. Its historical window spans **18 June 2017–17 June 2018**; the most recent 30 days are a development holdout, preceded by a 45-day gap. Five expanding CV folds have 30-calendar-day validation blocks and 45-day gaps. Figure `results/figures/09_validation_timelines.png` shows the corresponding dates and observed order counts for both tasks.

1. Construct the eligible historical data and buffered development holdout.
2. Search hyperparameters using the chronological CV folds. Classification maximises average precision (AP); regression minimises mean absolute error (MAE).
3. Compare default and tuned candidates on the development holdout. Classification selects by AP, then F1 and ROC-AUC to break ties; regression selects by MAE.
4. Refit candidates on the full eligible history and score the August cohort. Classification uses a threshold of 0.5.
5. Compare default and tuned candidates on the same August evaluation orders, then export the report tables.

Default configurations provide the within-family reference for tuning gains. The model chosen for August comes from the development holdout, not from August evaluation outcomes. Unknown outcomes are counted but excluded from performance metrics.

The optional June–August comparison is controlled by `RUN_MONTHLY_COMPARISON = False` in both notebooks. When enabled, it reruns CV tuning and model selection separately for each month and saves its outputs under `results/report_tables/monthly_comparison/`; pooled metrics are computed from individual predictions. The older optional regression April–July sensitivity analysis reuses June settings and should not be described as a fresh historical tuning exercise.

Published exports:

- [Classification tables](results/report_tables/classification_august_performance.html) and [LaTeX](results/report_tables/classification_august_performance.tex).
- [Regression tables](results/report_tables/regression_report.html) and [LaTeX](results/report_tables/regression_report.tex).

Open an HTML export in a browser and copy the required table into Word, retaining source formatting. The LaTeX files require `\usepackage{booktabs}`. Sample counts and exclusions are reported; orders with unknown outcomes are excluded from metrics.

Run each notebook from the first cell to regenerate August results and exports. The optional monthly comparisons take longer because they repeat the random searches. Final HTML/LaTeX exports and compact August summary CSVs are included in Git. Order-level prediction CSVs, extracted source CSVs, working report prose and Word documents stay local; the bundled source ZIP remains tracked.

## Data source

The source dataset is already tracked as `streamlit_release/data/olist_csv.zip`. The loader reads the extracted CSVs in `Olist_CSV/` and automatically extracts the bundled ZIP when that directory is absent. Extracted CSVs are Git-ignored to avoid duplicating the archive.

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
