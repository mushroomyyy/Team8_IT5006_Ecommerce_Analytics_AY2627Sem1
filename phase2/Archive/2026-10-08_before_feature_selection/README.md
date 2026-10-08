# Phase 2 — Modelling

Dual framing of one delivery-performance problem:
- **Classification** (`model_classification_dev.ipynb`): will an order arrive after its estimated delivery date?
- **Regression** (`model_regression_dev.ipynb`): by how many days will an order arrive before or after its estimated delivery date? (Lead time capped at 45 days.)

## Layout
| Path | Contents |
|---|---|
| `model_classification_dev.ipynb` | Classification development, June candidate selection and frozen July–August evaluation |
| `model_regression_dev.ipynb` | Days-from-promise development, June candidate selection and frozen July–August evaluation |
| `model_validation_timelines.ipynb` | Rebuilds the shared June development and candidate-selection timeline from actual eligible orders and saves `results/figures/09_validation_timelines.png` |
| `src/data.py` | Locates and loads the Olist CSVs (extracts the bundled zip on first run) |
| `src/features.py` | `build_feature_table` (v2's `final_df`), as-of seller/product history, `add_extra_features`, feature lists |
| `src/labels.py` | `get_required_dates`, `labels_as_of`, `regression_targets_as_of`, cohort evaluation helpers |
| `src/preprocessing.py` | `make_preprocessor`: median imputation, optional scaling and one-hot encoding, shared by both tracks |
| `src/splits.py` | `chronological_split`, `day_blocked_time_series_folds` |
| `src/evaluation.py` | Classification, regression and decile coverage metrics |
| `src/report_tables.py` | Shared export of formatted tables to Word-friendly HTML, booktabs LaTeX and CSV |
| `results/report_tables/` | June selection and frozen July–August report tables and summary CSVs; order-level predictions stay local |
| `src/report_figures.py` | Report figures for the regression notebook; each call draws inline and saves a PNG to `results/figures/` |
| `src/tuning.py` | Default classifier pipelines, search spaces, CV summary, random search, out-of-fold threshold selection |

Both tracks use a 365-day historical window, a 30-day development holdout preceded by a 45-day gap, and five expanding CV folds with 30-calendar-day validation blocks and 45-day gaps. Training outcomes must be known at each validation boundary. The final refit uses the full eligible historical window. Default and tuned candidates use identical rows within each comparison.

## June development and frozen-model evaluation

Both main notebooks use **2 June 2018** as the historical run date. The 365-day historical window spans **18 April 2017–17 April 2018**, with **19 March–17 April 2018** reserved as the 30-day development holdout. A 45-day gap precedes that holdout. Five expanding CV folds each validate on 30 calendar days after a 45-day gap; training outcomes must be known at each validation boundary. `results/figures/09_validation_timelines.png` shows the dates and eligible order counts for both tasks.

1. Evaluate starting configurations and tune hyperparameters using the same buffered historical training data. Classification maximises mean CV average precision (AP); regression minimises mean CV mean absolute error (MAE).
2. Compare starting and tuned candidates on the development holdout. Record the best-on-holdout candidate without treating it as the final deployment choice.
3. Refit all candidates on the full eligible June historical window and score June orders. Pool the daily June predictions to calculate candidate performance. Select the highest-AP classification candidate and lowest-MAE regression candidate using these June outcomes.
4. Discuss feature importance for the June-selected candidate. Score July and August with the same fitted pipeline, preprocessing and parameters, **without retraining, retuning or model reselection**.
5. Export compact candidate comparisons and frozen-model results with sample counts and explanatory notes.

Classification uses a threshold of 0.5. Default configurations provide the within-family reference for tuning gains. **June is a model-selection cohort, not an untouched final test**, because its outcomes determine the deployment candidate. July and August examine how that fixed model behaves on later orders.

**Timing interpretation:** assessment uses outcomes at approval + 45 days. The final June approvals reach that horizon on **14 August 2018** (available from 15 August under the strict-before-run convention). The July results therefore represent a retrospective frozen-model stress check, not a fully prospective claim that a model selected using all June outcomes was deployable on 1 July. August results likewise include orders before full June outcomes mature. These later cohorts are untouched by selection in this workflow, but their calendar timing must be reported accurately.

Open the exported classification and regression HTML files in `results/report_tables/` in a browser to copy tables into Word, retaining source formatting. Matching LaTeX exports use `booktabs`; compact CSVs contain the same reported metrics. Unknown outcomes are excluded from metrics and counted separately. The notebooks regenerate these outputs when run from the first cell.

Final HTML/LaTeX exports and compact summary CSVs are included in Git. Order-level prediction CSVs, extracted source CSVs, working report prose and Word documents stay local; the bundled source ZIP remains tracked.

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
- The development holdout contains the newest historical approval days. CV folds validate on dates after their training dates. June outcomes select the final candidate; July and August do not update it.

The classification notebook also exports the June-selected model’s full risk-decile table at the end: `results/report_tables/classification_june_deciles.html` (Word-friendly table), `.tex`, and `classification_june_decile_coverage.csv` (full-precision values). It reports individual-decile capture and lift, plus cumulative capture, using the same ranking population as the top-10% summary.
