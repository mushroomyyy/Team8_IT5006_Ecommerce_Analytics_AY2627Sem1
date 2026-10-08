# Common brief for notebook agents (simple-to-complex rebuild)

Repo: /Users/andrew_tjs/github/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/phase2 (branch `simple-to-complex`).
Python/Jupyter: ../.venv/bin/python, ../.venv/bin/jupyter. Run from phase2/.

## Read first
1. `PLAN_SIMPLE_TO_COMPLEX.md`: the authoritative plan (sections 0–9). Implement it.
2. The stage 1 library in `src/`, already built and tested (56 tests). Read the module docstrings: `datasets.py` (`build_task_data`: identical Train/Validation/history/folds everywhere), `linear_transforms.py`, `variable_selection.py`, `rf_search.py`, `regression_models.py`, `interpretation.py`, `inference.py`, `evaluation.py`, `tuning.py`, plus the existing `report_figures.py`, `report_tables.py` (`export_report_tables`), `validation_timelines.py` and `feature_selection.py`.
3. The previous notebooks, for reusable cells and figure ideas only: `model_classification_feature_selection.ipynb`, `model_regression_feature_selection.ipynb`, `model_validation_timelines.ipynb`, `classification_evaluation.ipynb`. Parse them with json.

## Decisions already made (do not reopen)
- Use the repo plan as written: C0/R0, C1/R1, C2/R2 (CV-stepwise + 1-SE), C2B/R2B (BIC-stepwise, supporting row), Lasso in the overlap table only, C3/R3 (RF, wide two-stage search). No winner is chosen. June is Test, July and August are Monitoring, and every model is scored everywhere.
- D6: plain logistic is unweighted (`src.tuning.plain_logistic`, lbfgs, tol=1e-8, max_iter=5000). RF tunes `class_weight`. Thresholded metrics are reported at the top-10% capacity cut-off and at 0.5. Also add ONE sensitivity row in classification: C1 with `class_weight='balanced'` (CV AP, ROC-AUC, Brier, top-10% precision). This shows weighting barely changes ranking but worsens calibration (or report whatever the numbers actually show).
- Quasi-separation: in Train, the 118 `route_type == '3. Mixed'` orders and the 227 orders with `seller_state_count >= 2` have ZERO late orders.
  - For ALL linear models in BOTH tasks, pass `merge_mixed_route=True` to the linear preprocessor (via `linear_kwargs` / `transform_kwargs`). It folds Mixed into All interstate; explain why in markdown.
  - Keep `seller_state_count`. In the logistic coefficient tables, flag terms with huge SEs as "not identified (quasi-separation: no late orders among multi-state-seller orders)". Use `max_se` / `exclude_terms` in `assert_coefficients_match`. State this in markdown.
  - RF uses the raw 24 features unchanged.
- RF hyperparameters: a background job is already writing the cache `results/simple_to_complex/{task}/rf_best_params.json` (plus the stage CSVs and a sensitivity figure). Use `RUN_RF_SEARCH = False` and `load_or_run_rf_search(task, run=False, ...)`. NEVER start another search. Before executing a task notebook, wait for the JSON to exist: poll every 60 s for up to 45 minutes. If it still isn't there, stop and report.

## User requirements (important)
- **Readable code: the repo may be graded.** Each notebook needs:
  - A title, a one-paragraph purpose, and a short contents list.
  - A markdown cell before every section saying what it does and why.
  - Short code cells, with heavy logic in `src/` (with docstrings).
  - No dead or commented-out code, no debug prints, consistent names, and the plan §2 terminology (Train, CV, Validation, Test (June), Monitoring) in all output, tables and figures.
- **Report diagrams must be generated and saved as PNG (dpi 200)** to `results/simple_to_complex/figures/`, with a numbered prefix per notebook (`01_…`, `02_…`, `03_…`). For example: the timeline/split diagram, the CV folds diagram, the stepwise path, the RF sensitivity plot, calibration, the decile lift chart, drift across June → July → August, residual diagnostics, PDPs and permutation importance. Before saving, make sure every figure has a title, axis labels with units, and a legend where needed. Don't use the default matplotlib style blindly: keep it clean and consistent, and reuse/extend `src/report_figures.py` styles.
- **Evaluation tables must exist:** risk-decile coverage → capture and lift (June, for every classifier), the full model comparison table (Train / CV mean ± SD / Validation / June / July / August), the success-criteria pass/fail table, overfitting gaps (Train − CV), and a drift/monitoring table plus a figure.
  - Export each with `export_report_tables` (HTML/TeX/CSV) to `results/simple_to_complex/{task}/`.
- Write each task's `run_metadata.json` to `results/simple_to_complex/{task}/run_metadata.json`. Include the dates, row counts per split, feature lists, transforms kept, selected variables, RF params, package versions and model fingerprints. Notebook 04 merges these.
- Leakage guards per plan §9: no LEAKAGE_COLS; selection, transforms and tuning see Train only; identical Validation and June order IDs across models; frozen fingerprints unchanged across June/July/August; statsmodels and scikit-learn coefficients agree.

## Rules
- Only edit the files your task assigns you. Treat every other `src/` file as read-only. If you hit a real bug in a read-only file, make the smallest fix and list it explicitly in your report.
- Don't commit, push, or touch `Archive/`. Don't delete or move the old notebooks; archiving happens later.
- Execute your notebook end-to-end from a clean kernel and keep the outputs:
  `../.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=-1`. Target under 25 minutes.
- Then run `../.venv/bin/python -m unittest` (it must stay green; add tests for any new `src` functions).
- Final report back: files created/changed, the section list of the notebook, every figure and table path, key numbers (headline metrics per model per split), any deviation from the plan, and anything surprising.

## Added later (user requirements)
- **Default vs tuned RF:** add an untuned RF row in each task: C3d/R3d, scikit-learn defaults with random_state=42, the same raw 24 features and preprocessor. Evaluate it everywhere C3/R3 is. Export a "tuning gain" table (tuned minus default on CV / Validation / June, plus Train − CV overfitting gaps) as `{task}_rf_tuning_gain`.
- **PR curve comparison (classification):** overlay the PR curves of the best plain logistic (C2, plus C1) and the tuned RF (C3) on June and on Validation. Show AP in the legend, a no-skill line at the late rate, and the top-10% operating point. Save as `figures/02_pr_curves.png`, with the helper in `src/model_figures.py`.
