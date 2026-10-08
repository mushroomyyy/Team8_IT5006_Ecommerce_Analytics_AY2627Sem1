# Code brief: simple → complex rebuild (Phase 2)

## Background (read first)

- **Project:** NUS IT5006 group project (Group 8), Milestone 2, *Analytics Implementation*. Data: the Olist Brazilian e-commerce dataset. Repo: `Team8_IT5006_Ecommerce_Analytics_AY2627Sem1`. All modelling code lives in `phase2/`.
- **Two prediction tasks per approved order**, using only information known at approval:
  - **Classification:** will the order arrive after its estimated (promised) delivery date? About 9–10% of orders are late.
  - **Regression:** by how many days will it arrive before or after the promised date? The lead time is capped at a 45-day waiting period.
- **What exists today:**
  - `src/` modules: `data`, `features`, `labels`, `splits`, `preprocessing`, `tuning`, `evaluation`, `feature_selection`, `report_tables`, `report_figures`, `validation_timelines`.
  - Current notebooks: `model_classification_feature_selection.ipynb` and `model_regression_feature_selection.ipynb` (the latest, and the source to restructure). Plus the older `model_*_dev.ipynb`, `classification_evaluation.ipynb` and `model_validation_timelines.ipynb`.
  - The current notebooks compare Logistic / Linear / Ridge, Decision Tree, Random Forest, XGBoost and LightGBM. They select a winner on June outcomes (LightGBM: June AP 15.09%, MAE 4.85 days) and run it frozen on July and August.
  - A multicollinearity screen (`python -m src.feature_selection`, documented in `MULTICOLLINEARITY_NOTES.md`) already reduces 41 candidate predictors to **24 raw features**, which become 53 encoded columns.
- **What the professor asked for:** show the progression from simple to complex. Build the simple (plain linear / logistic) models *properly* first, with careful variable selection and diagnostics, then a more complex model. Don't present a seven-model leaderboard.
- **What happens after this brief:** once the code is done, a different agent rewrites the Word report from `RESULTS_SUMMARY.md` and the exported tables and figures. So outputs must be complete and self-explanatory.

---

**Audience:** the coding agent. This brief covers **code and notebooks only**. A separate agent will update the Word report afterwards from the outputs in §9. Do not edit the report.

**Why:** Prof feedback. The current pipeline compares seven model families and picks a winner. The new story: build the **plain linear models properly first** (OLS and logistic regression, with transformations and CV-judged variable selection). Then ask whether a harder model (Random Forest) adds enough to justify itself. **Nothing is "chosen".** Every model is evaluated the same way, and June is the real evaluation.

---

## 0. Decisions (confirmed unless marked)

| # | Decision | Setting |
|---|---|---|
| D1 | Model families | **Linear (OLS / plain logistic) + Random Forest only.** Decision Tree, XGBoost and LightGBM are dropped from the new notebooks; old results stay archived. |
| D2 | Linear models are *plain* | Regression: **OLS** (`LinearRegression`). Classification: **unpenalised logistic** (`LogisticRegression(penalty=None)`). No Ridge, no hyperparameters. For linear models, CV is used for variable selection and honest performance estimates, not tuning. |
| D3 | Primary metrics | **Classification: AP. Regression: RMSE.** All other metrics are still reported. RF tuning and the stepwise criterion use these. |
| D4 | Variable selection | **Primary: forward stepwise selection scored by chronological CV + 1-SE rule.** **Cross-check 1:** information-criterion stepwise (BIC as the rule, AIC also reported) on Train, with a supporting evaluated row. **Cross-check 2 (light):** Lasso / L1-logistic, used only to add a column to the overlap table; no model row, no plots. Categoricals move as whole blocks. |
| D8 | Random Forest tuning | **Wide, two-stage search** (§4.6), scored by AP / neg RMSE on the same chronological folds. Search results are cached so notebooks don't rerun them by default. |
| D5 | No model selection | All models get the same evaluation table: Train (in-sample), CV mean ± SD, Validation, **June (Test, the real evaluation)**, July and August (Monitoring). |
| D6 | Class weights *(confirm)* | Plain logistic **unweighted** (meaningful probabilities, standard odds ratios). RF: `class_weight` is part of its tuning space. Thresholded metrics are reported at the **top-10% capacity** cut-off (the same operating point for every model), plus at 0.5 for reference. |
| D7 | Transformations | **Linear models only**, decided on Train: log1p for skewed inputs, sine/cosine for month and weekday, an optional squared term after a curvature check. Each kept only if CV improves (§4.2). RF uses the raw 24 features. |

---

## 1. What stays exactly the same (do not change)

- Run date `2018-06-02`; `PXD = 365`; `LOOKBACK` / `WAITING_PERIOD = 45`; Validation = 30 days after a 45-day gap.
- CV folds: `day_blocked_time_series_folds(n_splits=5, gap_days=45, validation_days=30)` followed by `available_outcome_folds`.
- Labels and targets: `labels_as_of`, `regression_targets_as_of`, `attach_prediction_labels`, `attach_regression_actuals` (`src/labels.py`). The 45-day cap and `clip_to_waiting_period`.
- The upstream multicollinearity screen: 24 raw features (`SELECTED_NUM_COLS` + `SELECTED_CAT_COLS`), reference coding via `make_preprocessor(..., reference_categories=True)`. Variable selection happens **after** this, inside the 24.
- Leakage rules in `README.md`. `RANDOM_STATE = 42`. Daily June inference via the existing `run_inference` pattern, with predictions pooled before metrics.
- Reference baselines: classification `DummyClassifier(strategy='prior')`. Regression `DummyRegressor(median)` and `PromiseBaseline` ("Olist promise").

## 2. Terminology (use in all code output, tables and figures)

| Name | What it is |
|---|---|
| **Train** | Buffered development-training rows (`train_df`), approved before the 45-day gap preceding Validation, with outcomes known at Validation start. "In-sample" means a model fitted and scored on Train. |
| **CV** | 5 expanding chronological folds inside Train. |
| **Validation** | 30-day out-of-time holdout (19 Mar – 17 Apr 2018). Used for diagnostics and discussion only; no choices are made on it. |
| **Test (June)** | June 2018 orders, scored daily by models refit on the full eligible history before 2 June. **The headline evaluation.** |
| **Monitoring** | July and August, scored by the same frozen June-fitted models (every model, not one winner). |

## 3. Protocol (per task)

1. **Transformations (linear only):** decide on Train via CV (§4.2).
2. **Linear, all variables** (C1 / R1): CV on Train; fit on Train.
3. **Forward stepwise + 1-SE** on Train CV → selected subset → **Linear, selected** (C2 / R2).
4. **Cross-checks** on Train: BIC-stepwise subset → **Linear, BIC-selected** (C2B / R2B, supporting row). Lasso → selected set for the overlap table only.
5. **Random Forest:** randomised hyperparameter search on Train CV (AP / RMSE) → fit on Train.
6. Score every model on **Train (in-sample)** and **Validation**.
7. **Refit** every model's fixed specification (same transforms, same selected variables, same RF hyperparameters) on the full eligible history before 2 June → score **June**.
8. Freeze each fitted June model (record a fingerprint) → score **July** and **August**. No refit, retune, reselection or re-transform.

Every decision (transforms, variables, RF hyperparameters) uses Train only. Assert in code that Validation, June, July and August rows never enter steps 1–5.

## 4. Models and methods

### 4.1 Model list
| Key | Classification (late delivery) | Key | Regression (days from promise) |
|---|---|---|---|
| C0 | Dummy (prior): no-skill | R0a / R0b | Dummy (median); Olist promise |
| C1 | Logistic, plain, all 24 (+ D7 transforms) | R1 | OLS, all 24 (+ D7 transforms) |
| C2 | Logistic, plain, **CV-stepwise subset** | R2 | OLS, **CV-stepwise subset** |
| C2B | Logistic, plain, BIC-stepwise subset *(supporting)* | R2B | OLS, BIC-stepwise subset *(supporting)* |
| C3 | Random Forest, tuned (AP), raw 24 | R3 | Random Forest, tuned (neg RMSE), raw 24 |

- RF: wide search per §4.6 (replaces the old small spaces).
- Unpenalised logistic with 26 state dummies may hit quasi-separation for rare states. Use `solver='lbfgs'`, `max_iter=5000`, and check convergence warnings. If it fails, record that and fall back to a negligible L2 (`C=1e6`), stated explicitly.

### 4.2 Transformations for the linear models (new; goes in the report's Feature Engineering section)
Implement as options in `make_preprocessor` (or a linear-only preprocessor). Everything is fitted inside the pipeline on training rows only.
1. **log1p for skewed non-negative numerics.** Candidates: `payment_value_sum`, `order_frieght_value_sum`, `order_product_weight_g_sum`, `order_product_volume_cm3_sum`, `approval_lag_hours`, `seller_dist_km_mean`, `freight_to_price_ratio`, `orders_approved_prev_7d`. Compute skewness on Train; transform if |skew| > 1. Save the skewness table (before and after).
2. **Cyclic calendar:** replace `order_approved_month` and `order_approved_day_of_week` with sine/cosine pairs for the linear models. Keep `order_approved_day_of_month` numeric, or check whether sine/cosine helps. Keep the weekend, Black Friday and December flags. Dummies would be exactly collinear with the flags, which is why sine/cosine is used.
3. **Curvature check:** component-plus-residual (partial residual) plots for the top 3 numeric predictors of R1 / C1. If curvature is clear, add a squared term for that predictor only.
4. **CV comparison:** C1 / R1 with raw inputs vs with each transform group added (log, cyclic, squared). Keep a group only if mean CV AP / RMSE improves. Save `transform_cv.csv`.
5. **Re-run the VIF** on the final linear design (reuse `numeric_vif` from `src/feature_selection.py`). Save `linear_design_vif.csv`.

### 4.3 Forward stepwise with CV: new `src/variable_selection.py`
- **Units:** raw features. The 3 categoricals are single units (whole dummy blocks). Transformed variants travel with their raw feature (e.g. log weight replaces weight; month sin and cos enter together).
- **Algorithm:**
  1. Start empty (intercept only).
  2. For each step, try adding each remaining unit. Score the pipeline with `cross_validate` on the 5 chronological folds (AP, or neg RMSE). Record the mean and SE across folds for every candidate.
  3. Add the best unit. Continue until all units are in. That gives the **full path**.
- **Stopping: 1-SE rule.** Choose the smallest step whose mean CV score is within 1 SE of the best step on the path. Also record the best-score step.
- **Outputs:**
  - `stepwise_path.csv`: step, added unit, n units, n encoded columns, CV mean, CV SE.
  - `stepwise_candidates.csv`: every candidate tried at every step.
  - The selected list.
  - A path plot (CV score vs step, ±1 SE, chosen step marked).
- **Cost:** 300 candidate evaluations × 5 folds. Use `n_jobs=-1`. If too slow, allow early stop after 4 consecutive non-improving steps, but log it.
- **Explanation for the report:** each addition is judged on later, unseen time periods, not on in-sample p-values, which limits the usual stepwise overfitting (Harrell). The remaining optimism is checked by Validation and June.

### 4.4 Information-criterion stepwise cross-check (same module)
- Same units, same forward procedure, but each candidate is scored by the **in-sample information criterion on Train**, not by CV.
  - OLS: `statsmodels.OLS`. Logistic: `statsmodels.Logit` (unweighted).
  - AIC = 2k − 2 ln L; BIC = k ln(n) − 2 ln L, where k counts encoded columns, so a categorical block costs all its dummies.
- Add the unit that lowers **BIC** most. Stop when no addition lowers BIC. Also run the AIC version and report its selected set (AIC is usually larger, because BIC penalises more).
- **Outputs:** `ic_path.csv` (step, unit, k, AIC, BIC); the selected sets.
- **Why both CV and IC:** IC is the classical statistics approach. It's fast, needs no folds, and is based on likelihood. But it assumes independent rows and a stable relationship over time. CV-stepwise judges every addition on *later* time periods, which matches how the model is used. Agreement between them is evidence of robustness; disagreement is worth a sentence in the report.
- Logistic quasi-separation: if `Logit` fails to converge for a candidate, record that and skip the candidate at that step (log it).

### 4.5 Lasso cross-check (keep it light)
- One short function: `LassoCV` / `LogisticRegressionCV(penalty='l1', solver='saga')` on the same linear design and the same folds, using default penalty grids. Take the CV-best penalty, fit on Train, and map non-zero coefficients to raw units (a categorical block is kept if any dummy is non-zero). No 1-SE rule, no separate model, no plots. Its only output is a column in the overlap table.
- **Overlap table (`selection_overlap.csv`):** each unit × {CV-stepwise selected (and the step it entered), BIC selected, AIC selected, Lasso selected}, plus a count of how many methods chose it.
- Fit the plain model on the BIC subset (C2B / R2B) as a supporting row. Lasso gets no model row.

### 4.6 Random Forest: wide two-stage search (new `src/rf_search.py`)
- **Stage A (broad):** `RandomizedSearchCV`, `n_iter = 60`, the same 5 chronological folds, `refit=False`, `n_jobs=-1`, `n_estimators` fixed at 200 for speed.
  - **Shared:** `max_depth ∈ {None, 6, 8, 10, 12, 16, 20, 30}`; `min_samples_leaf ∈ {1, 2, 5, 10, 20, 50, 100, 200}`; `min_samples_split ∈ {2, 5, 10, 20, 50}`; `max_features ∈ {'sqrt', 'log2', 0.2, 0.3, 0.5, 0.7, 1.0}`; `max_samples ∈ {None, 0.5, 0.7, 0.9}` (`bootstrap=True`).
  - **Classifier only:** `class_weight ∈ {None, 'balanced', 'balanced_subsample'}`; `criterion ∈ {'gini', 'entropy'}`.
  - **Regressor:** `criterion = 'squared_error'` (matches RMSE).
- **Stage B (refine):** a small grid around the Stage A best: neighbouring values of `max_depth`, `min_samples_leaf` and `max_features`, with `n_estimators ∈ {300, 500}`. Pick the best mean CV score; the tie-break goes to the simpler model (shallower depth / larger leaves).
- **Outputs:**
  - `rf_search_stageA.csv` and `rf_search_stageB.csv`: all candidates with mean and SD CV scores and fit times.
  - `rf_best_params.json`.
  - A sensitivity plot: CV score vs `max_depth` and vs `min_samples_leaf`, from the Stage A results.
- **Caching:** the search runs from a script (`python -m src.rf_search --task classification|regression`) or behind a notebook flag `RUN_RF_SEARCH = False`. When false, it loads the cached JSON / CSVs. Record the run time and package versions in `run_metadata.json`.
- **Budget:** if Stage A is too slow on a laptop, reduce `n_iter` to 40 and log the change. Don't shrink the space.

### 4.7 Guards and tests
- Selection functions take Train rows and folds only, and raise if handed extra rows. Unit tests (`test_variable_selection.py`):
  - Toy data where one feature is pure signal: forward selection (CV and BIC) picks it first.
  - BIC stops when the additions are pure noise.
  - Categorical blocks are added or removed whole.
  - The 1-SE rule picks the smallest qualifying step.
  - Deterministic with `RANDOM_STATE`.
  - Raises on non-Train rows.

## 5. Evaluation (every model, same table)

**Columns:** Train (in-sample) | CV mean ± SD | Validation | **June** | July | August.

- **Classification:**
  - AP (primary), ROC-AUC, Brier score.
  - Precision / recall / F1 at the top-10% cut-off and at 0.5.
  - Top-10% capture and lift; full decile table (`risk_decile_coverage`) for June.
- **Regression:**
  - RMSE (primary), MAE, R², median AE, bias.
  - RMSE and MAE split by actual late vs on-time.
  - Derived late flag (prediction > 0) → precision / recall / F1 vs the classification label (dual framing).
- **Success criteria table** (value + pass/fail, on June; also shown on Validation):
  - Classification: top-10% lift ≥ 2. RF top-10% precision ≥ 1.10 × best plain logistic (C1/C2).
  - Regression: RMSE ≥ 10% below the median baseline, and below the Olist promise. RF RMSE vs best OLS (report the relative gain).
- **Overfitting gap:** Train minus CV for each model (expect a large gap for RF). Export it explicitly.
- **Monitoring:** June → July → August for every model; drift plot.

## 6. Model interpretation (new section; its own notebook part)

- **Linear models (C1, C2, R1, R2):** refit with `statsmodels` on Train (same design) to get coefficients with 95% CIs.
  - OLS uses robust **HC3** standard errors.
  - Logistic reports odds ratios with CIs. Give both per-unit (raw scale; for log inputs, "per doubling") and per-1-SD.
  - Categorical effects are read relative to their reference (SP, `1. All interstate`, single-method `credit_card`).
  - Assert that the statsmodels and scikit-learn coefficients match (tolerance) for the same unweighted spec.
- **Hypothesis check table:** the six feature-engineering hypotheses (time, workload, order, distance, payment, promise), the expected sign, the observed sign / significance in C2 and R2, and agree / disagree.
- **Diagnostics:**
  - OLS: residual vs fitted, Q-Q / residual histogram, residual by route type and top states; note heteroscedasticity (why HC3).
  - Logistic: calibration curve on Validation and June.
- **Random Forest:**
  - Permutation importance on Validation and June (raw features, primary metric).
  - Partial-dependence plots for the top 3 features.
  - Agreement table: RF top-10 features vs linear significant coefficients.

## 7. Notebooks: cut down to this set

Archive first (§8). Then:
1. **`01_data_features_multicollinearity.ipynb`**
   - Data load and cleaning summary.
   - Feature table.
   - Label and target definitions plus the waiting-period audit.
   - Multicollinearity (moved from `*_feature_selection.ipynb`).
   - **Transformations for linear models** (§4.2 skewness, cyclic and VIF parts; the CV comparison may live in 02/03).
   - Train / CV / Validation / Test / Monitoring timeline figure (absorbs `model_validation_timelines.ipynb`).
2. **`02_classification.ipynb`**: setup → split → C0 → Stage 1 linear (transforms CV → C1 → stepwise → C2 → Lasso → C2L) → Stage 2 RF (C3) → evaluation table → June deciles → monitoring → interpretation → exports.
3. **`03_regression.ipynb`**: the same structure with R0–R3 and the dual-framing diagnostic.
4. **`04_summary`** (notebook or `python -m src.summary`): cross-task "simple → complex ladder" figures and `RESULTS_SUMMARY.md`.

Remove:
- The legacy quick-fit "Model Training / Evaluation – Test Set / Backtesting" block in the classification notebook.
- "How much does a random split flatter the scores?"
- All DT / XGB / LGBM code.
- The development-holdout "winner" and "June-selected" logic.
- `classification_evaluation.ipynb` (archive it).

Move into `src/`:
- `variable_selection.py` (new: CV-stepwise, IC-stepwise, Lasso, overlap).
- `rf_search.py` (new: two-stage search, caching, sensitivity plot).
- `regression_models.py` (new: `PromiseBaseline`, `clip_to_waiting_period`, builders, RF space).
- `linear_transforms.py` (new) or `make_preprocessor` options.
- `interpretation.py` (new: statsmodels tables, hypothesis check, PDP helpers).
- Slim `tuning.py` to Logistic + RF + Dummy.
- One shared `cv_summary` (mean and SD, any scoring dict).

## 8. Housekeeping

- New git branch `simple-to-complex`; small commits. **Do not push or merge without asking.**
- Archive to `Archive/2026-10-09_before_simple_to_complex/`: the current notebooks + `src/`, with `snapshot_manifest.json` (same pattern as the 2026-10-08 archive). Don't delete existing `results/`.
- All new outputs go under `results/simple_to_complex/{classification,regression,figures}/`, plus `RESULTS_SUMMARY.md` and `run_metadata.json` (dates, row counts, feature lists, transforms kept, selected variables, RF params, package versions, model fingerprints). Use `export_report_tables` for HTML / TeX / CSV.
- Add `statsmodels` to `requirements.txt`. Leave xgboost / lightgbm / catboost until nothing imports them.
- Update `README.md` (layout, protocol: no selection, June is Test, model list).
- Aim for each notebook to run end-to-end in under ~25 min on a laptop.

## 9. Acceptance checks

- [ ] `python -m unittest` passes (existing tests updated + `test_variable_selection.py`).
- [ ] `jupyter nbconvert --execute` succeeds for 01–03 (and 04) from a clean kernel.
- [ ] Assertions:
  - No `LEAKAGE_COLS`.
  - Selection, transform and tuning steps receive Train rows only.
  - All models are scored on identical Validation / June order IDs and labels.
  - Frozen fingerprints are unchanged across June, July and August.
  - statsmodels and scikit-learn coefficients agree.
- [ ] `RESULTS_SUMMARY.md` lists, per task:
  - Row counts per split.
  - Transforms kept (with CV evidence).
  - Stepwise path summary, chosen step, and selected variables.
  - The three-method overlap (CV-stepwise / BIC / AIC / Lasso).
  - RF search summary (space, n_iter, best params, CV score) and the sensitivity findings.
  - The full evaluation table (Train / CV / Validation / June / July / August).
  - Success-criteria pass/fail.
  - Overfitting gaps.
  - Top coefficients / odds ratios with CIs.
  - Hypothesis-check table.
  - RF top features.
  - A "what changed vs the previous pipeline" note (previous headline: LightGBM June AP 15.09%, MAE 4.85 d).

## 10. Notes for the report phase (not for coding)

- Section order:
  1. Business context
  2. Data
  3. Feature engineering + multicollinearity + transformations for linear models
  4. Evaluation design (train / CV / validation / test / monitoring)
  5. Metrics (AP, RMSE) + success criteria
  6. Model descriptions
  7. Stage 1 linear (all → CV-stepwise; BIC and Lasso checks)
  8. Stage 2 RF
  9. Results + drift
  10. Model interpretation
  11. Insights
  12. Limitations
- RF "variable selection" wording: RF does not remove variables. Split choice is embedded selection, and RF is robust to uninformative inputs. Cite Breiman (2001) ML 45:5–32; Guyon & Elisseeff (2003) JMLR 3:1157–1182; Genuer, Poggi & Tuleau-Malot (2010) PRL 31(14):2225–2236. Caveat: Strobl et al. (2007) BMC Bioinformatics 8:25 (importance bias). Verify the details before citing.
- Information criteria: Akaike (1974) IEEE Trans. Automatic Control 19(6):716–723; Schwarz (1978) Annals of Statistics 6(2):461–464.
- Selection: stepwise criticism, Harrell, *Regression Modeling Strategies*; Lasso, Tibshirani (1996) JRSS-B 58(1):267–288; 1-SE rule, Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009), §7.10.
- Trees are invariant to monotone transforms: one sentence explaining why the transforms are linear-only.
- Remove the old limitation "June outcomes informed model selection" (no longer true). Resolve the regression class-imbalance placeholder (no reweighting in regression).
