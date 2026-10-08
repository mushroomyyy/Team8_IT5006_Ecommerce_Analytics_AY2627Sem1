# Simple → complex rebuild: status and handoff

**For:** whoever continues this work if the current session stops.
**Last updated:** 2026-10-09, after notebook 02 and the archiving. Notebook 03 was in progress at the time.

## Goal
Implement `phase2/PLAN_SIMPLE_TO_COMPLEX.md` (the repo copy is authoritative) on branch `simple-to-complex`. When every acceptance check in plan §9 passes, **merge into `main` and push** (the user approved this). Phase 2 should end up holding only what the new Word report uses. Everything else is moved to `Archive/` so it is easy to restore. After this work, the only remaining job should be writing the report.

## Decisions made with the user
- **Plan version.** Use the repo plan:
  - BIC-stepwise supporting row (C2B/R2B);
  - Lasso only in the overlap table;
  - wide two-stage RF search (§4.6).
- **D6 (class weights).**
  - Plain logistic is unweighted (`src.tuning.plain_logistic`).
  - RF tunes `class_weight`.
  - Thresholded metrics are reported at the top-10% cut-off and at 0.5.
  - Add one CV sensitivity row for balanced logistic.
- **Class-imbalance experiment (classification only).**
  - Compare random undersampling, random oversampling and SMOTENC on C1 and C3. Resampling must happen inside CV training folds only, via `imblearn.pipeline.Pipeline`.
  - Report the results; do not adopt any of them. If one clearly improves CV AP, ask the user before adopting it.
  - Cite van den Goorbergh et al. (2022) JAMIA 29(9) and Chawla et al. (2002) JAIR 16, verifying both before citing.
- **Untuned RF baseline.** Add C3d/R3d: scikit-learn defaults with `random_state=42`, the same raw 24 features.
  - Evaluate it everywhere C3/R3 is evaluated.
  - Export a `{task}_rf_tuning_gain` table: tuned minus default on CV, Validation and June, plus the Train − CV overfitting gap for both.
- **Quasi-separation.**
  - In Train, the 118 `route_type == '3. Mixed'` orders and the 227 orders with `seller_state_count >= 2` have zero late orders.
  - Linear models pass `merge_mixed_route=True` (`src/linear_transforms.py`).
  - `seller_state_count` is kept and flagged "not identified" in the logistic tables.
  - RF uses the raw features.
- **Readability.** Code must be readable because the repo may be graded:
  - each notebook has a title, a purpose and a contents list;
  - every section has a markdown intro;
  - code cells are short and heavy logic lives in `src/`;
  - no dead code.
- **Figures and tables.** Every report diagram and evaluation table must be generated:
  - CV/timeline diagrams;
  - deciles → capture and lift;
  - the model comparison table;
  - the success-criteria table;
  - overfitting gaps;
  - drift across June → July → August.
- **Subagents.** Use cheap (Sonnet) subagents for the implementation; the main session validates their work.

## Key facts and numbers so far
- **Row counts.** Train is 45,972 (classification) and 45,371 (regression). Validation is 6,870 and 6,847. History before 2 June is 63,592 and 62,841.
- **Late rates.** Train 8.83%, Validation 10.38%.
- **Linear transforms (notebook 01).** All 8 log1p candidates have |skew| > 1, so all are logged; month and weekday are made cyclic.
  - The linear design has 54 encoded columns.
  - The largest VIF after the log transform is 6.4 (`log_order_frieght_value_sum`, which is nearly collinear with `log_payment_value_sum` and `log_freight_to_price_ratio`). This is acceptable (below 10). The CV comparison in 02/03 decides whether the log group is kept; mention it in the report.
- **RF cache** (`results/simple_to_complex/{task}/rf_best_params.json`, `rf_search_stageA.csv`, `rf_search_stageB.csv`, sensitivity figures):
  - classification best CV AP is 0.2691 ± 0.0331;
  - regression best CV RMSE is 8.335 ± 0.819.
  - Notebooks load the cache with `RUN_RF_SEARCH = False`. Rerunning takes about 10 minutes per task: `../.venv/bin/python -m src.rf_search --task <task>`.
- **Previous pipeline headline** (for the "what changed" note): LightGBM, June AP 15.09%, MAE 4.85 days.

## Rules for the implementation agents
The full brief is `REBUILD_AGENT_BRIEF.md`; give it to every agent.

**File ownership** (avoids agents overwriting each other):
- 01 owned `report_figures.py`, `validation_timelines.py`, `feature_selection.py`, `data_audits.py` and the notes.
- 02 owns `evaluation.py`, `model_figures.py` and `resampling.py`.
- 03 should own `regression_models.py` and reuse `model_figures.py`, adding to it only after 02 is committed.

**Running things:**
- Execute a notebook: `../.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=-1`, from `phase2/`.
- Tests: `../.venv/bin/python -m unittest`, from `phase2/`.

**Committing results:** the repo `.gitignore` ignores `*.csv`. The `results/simple_to_complex/**` folders have `.gitignore` files containing `!*.csv`.

## Progress
| # | Step | Status |
|---|---|---|
| 1 | Commit the multicollinearity screen and notes | done (504e4a0) |
| 2 | Archive notebooks and `src/` → `Archive/2026-10-09_before_simple_to_complex/` (with `snapshot_manifest.json`) | done (05af09e) |
| 3 | Stage 1 `src/` library plus unittest suite | done (a84a80f) |
| 4 | Packages in `.venv` (statsmodels, nbconvert, pytest, imbalanced-learn) and `requirements.txt` | done |
| 5 | RF search caches for both tasks | done (committed alongside this file) |
| 6 | `01_data_features_multicollinearity.ipynb`, executed | done (7021ebf) |
| 7 | `02_classification.ipynb` (including the resampling experiment, C3d and PR curves) | done (e7f4a89) |
| 8 | `03_regression.ipynb` | **in progress** (agent started after 02); if it is missing or incomplete, use the step 8 task below |
| 9 | `04_summary` → `RESULTS_SUMMARY.md` plus merged `run_metadata.json` | to do |
| 10 | Clean-up: archive the old notebooks and results, rewrite README | archiving done (a82cb48, 2789736, 34cd201); README rewrite still to do |
| 11 | Acceptance checks (plan §9) | to do |
| 12 | Merge into `main` and push | to do |

## Resuming step 7 (classification)
1. If `02_classification.ipynb` exists but is uncommitted, it is partial agent work. Check whether it has outputs and no error cells, using this one-liner from `phase2/`:
   `../.venv/bin/python -c "import json;nb=json.load(open('02_classification.ipynb'));c=[x for x in nb['cells'] if x['cell_type']=='code'];print(sum(x.get('execution_count') is None for x in c),'unexecuted;',sum(o.get('output_type')=='error' for x in c for o in x.get('outputs',[])),'errors')"`
2. If it is incomplete, give a fresh agent `REBUILD_AGENT_BRIEF.md` plus this task: *"Finish `02_classification.ipynb` per plan §3–§7 item 2, including the class-imbalance experiment and the C3d default-RF row with the tuning-gain table; you own `src/evaluation.py`, `src/model_figures.py` and `src/resampling.py`."*
3. Validate: the notebook executes cleanly; figures `results/simple_to_complex/figures/02_*.png` exist; tables are in `results/simple_to_complex/classification/`; unittest passes.
4. Commit only 02's files.

## Step 8 (regression): agent task
Give an agent `REBUILD_AGENT_BRIEF.md` plus:
- Create `03_regression.ipynb` with the same structure as 02. Models: R0a Dummy (median), R0b Olist promise, R1, R2 (CV-stepwise), R2B (BIC-stepwise), R3 (tuned RF), and R3d (default RF).
- Add the dual-framing diagnostic: a derived late flag (prediction > 0) compared with the classification label.
- Include RMSE/MAE split by late vs on-time, and OLS diagnostics with HC3.
- Reuse `src/model_figures.py` and the `src/evaluation.py` helpers. The agent owns `src/regression_models.py`.
- Primary metric is RMSE. Use `merge_mixed_route=True` for the linear models. Load the RF cache.
- Outputs go to `results/simple_to_complex/regression/` and figures to `figures/03_*.png`.

## Step 9 (summary)
Run `04_summary.ipynb` or `python -m src.summary`. It should:
- read both tasks' exported tables and `run_metadata.json` files;
- produce the cross-task "simple → complex ladder" figures (`figures/04_*.png`);
- write `results/simple_to_complex/RESULTS_SUMMARY.md` with everything listed in plan §9, plus the resampling result, the tuning gain, and the "what changed" note.

`RESULTS_SUMMARY.md`, `MULTICOLLINEARITY_NOTES.md` and the exported tables and figures are the inputs the report writer uses.

## Step 10 (clean-up)
Copies of the following already exist in `Archive/2026-10-09_before_simple_to_complex/`. Remove them from `phase2/` with `git rm`:
- `classification_evaluation.ipynb`
- `model_classification_dev.ipynb` and `model_regression_dev.ipynb`
- `model_classification_feature_selection.ipynb` and `model_regression_feature_selection.ipynb`
- `model_validation_timelines.ipynb`

Move old result folders that the new notebooks don't produce into `Archive/2026-10-09_before_simple_to_complex/results/` with `git mv`: everything under `results/` except `feature_selection/` and `simple_to_complex/`, including `feature_selection_run/`.

Also:
- Move `PLAN_SIMPLE_TO_COMPLEX.md`, `REBUILD_STATUS.md` and `REBUILD_AGENT_BRIEF.md` into the archive folder at the very end.
- Rewrite `README.md`: the layout; the protocol (no model selection, June is Test, July and August are Monitoring); the model list; how to run 01–04 and the tests; and where outputs live.
- Remove the xgboost, lightgbm and catboost entries from `requirements.txt` if `grep -r "xgboost\|lightgbm\|catboost" src *.ipynb` finds nothing.

## Step 11 (acceptance)
1. `../.venv/bin/python -m unittest` is green.
2. Execute 01, 02, 03 and 04 with nbconvert from a clean kernel; each should take under about 25 minutes.
3. Spot-check the plan §9 assertions in the notebook outputs:
   - no leakage columns;
   - Train-only selection;
   - identical Validation and June order IDs across models;
   - fingerprints unchanged across months;
   - statsmodels and scikit-learn coefficients agree.

## Step 12 (merge and push)
`git switch main && git merge --no-ff simple-to-complex && git push origin main`. Pushing the branch too is fine.

## Notebook 02 key results (for the report and the summary)
- **No resampling adopted.** It never improves CV AP and badly worsens Brier: C1 Brier 0.087 → 0.255–0.275.
- **Balanced logistic:** CV AP 0.137 vs 0.140 unweighted, and Brier 0.280 vs 0.087.
- **June AP:** C1 8.2%, C2 6.8%, C2B 8.1%, C3 11.2%, C3d 11.0%; the no-skill rate is 2.2%.
  - June has only 137 late orders, a 2.2% late rate. The old pipeline's June evaluation had the same late count, and its LightGBM June AP was 15.09%.
- **RF vs logistic:**
  - CV AP: RF 0.27 vs logistic 0.14–0.15.
  - On Validation and June the RF is not clearly better; it fails the "1.10 × logistic top-10% precision" criterion.
  - The RF drifts worst: August ROC-AUC 0.43.
- **C2 (CV-stepwise, 1-SE)** keeps 6 features: customer_state, day_of_month, seller_count, day_of_week, promised_lead_days, route_type. BIC keeps 14, AIC 19, Lasso all 24.
- **Tuning gain** (tuned C3 vs default C3d): +0.017 CV AP, +0.016 Validation AP, +0.003 June AP. The default forest's overfitting gap is huge: Train AP 100%.

## Pending user request (do after 03 is committed)
- **ROC curves in notebook 02.** Add a ROC curve figure (TPR vs FPR; top-left is best; chance diagonal) for C1, C2 and C3 on Validation and June. Put ROC-AUC in the legend and mark the top-10% operating point.
  - Add the helper to `src/model_figures.py` next to the PR-curve helper, and save the figure as `figures/02_roc_curves.png`.
  - Re-execute 02 (about 5 minutes) and commit.
  - Wait until the 03 agent is finished, because it may also be editing `src/model_figures.py`.
- **No LightGBM in the modelling notebooks.** Remove the `PREVIOUS_PIPELINE` constant and the "versus previous pipeline (LightGBM)" print from 02 and 03, and from their `run_metadata.json`. Keep the old-pipeline comparison (June AP 15.09%, MAE 4.85 days) only in 04 / `RESULTS_SUMMARY.md`, as the "what changed" note. Do this after 03 is committed, together with the ROC change, then re-execute 02 and 03.
