# Report handoff: simple → complex rebuild

**For:** the agent updating the Word report (e.g. Claude desktop). Read local files only.
**Repo root:** `/Users/andrew_tjs/github/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/phase2/`, on branch `simple-to-complex` until it is merged into `main`.
**Status (2026-10-09):** sections 1–8 below can be drafted now. The regression results are final once notebook 03 is committed. `RESULTS_SUMMARY.md` (the one-stop summary) does not exist yet; when it appears in `results/simple_to_complex/`, re-check every number against it.

## Ground rules
- Every number must come from the files listed below. Don't invent or round differently from the tables.
- Use these terms throughout: **Train**, **CV** (5 chronological folds inside Train), **Validation** (30-day out-of-time holdout, 19 Mar–17 Apr 2018, diagnostic only), **Test (June)** (the headline evaluation) and **Monitoring** (July and August, scored by frozen models).
- **No model is "selected" or called the "winner".** Every model is evaluated the same way, and the story runs from simple to complex.
- The models:
  - Classification: C0 no-skill; C1 logistic with all 24 features; C2 logistic with CV-stepwise features; C2B logistic with BIC-stepwise features; C3 tuned Random Forest; C3d default Random Forest.
  - Regression: R0a median baseline; R0b Olist promise baseline; R1 OLS; R2 OLS with CV-stepwise features; R2B OLS with BIC-stepwise features; R3 tuned Random Forest; R3d default Random Forest.
- Primary metrics: **AP** for classification (write "AP, the area under the precision-recall curve" once) and **RMSE** for regression.
- Delete the old pipeline's text: XGBoost, LightGBM, CatBoost, the ensemble, "June-selected" models, the paired model comparison, and the limitation "June outcomes informed model selection".

## Still changing: keep monitoring these
Draft freely, but re-check the items below before finalising. Run `git log --oneline -5` in the repo (or look at the progress table in `REBUILD_STATUS.md`) to see what has landed.

| Item | What may change | Watch for |
|---|---|---|
| **Regression results** (`results/simple_to_complex/regression/*`, `figures/03_*`) | The notebook has run but has not been reviewed or committed yet. Numbers could change if the review finds a bug | The commit "Add notebook 03…" |
| **ROC curves** (`figures/02_roc_curves.png`) | Not created yet | The file appears |
| **LightGBM comparison** | It moves out of notebooks 02/03 and their `run_metadata.json` into `RESULTS_SUMMARY.md` | Take it from `RESULTS_SUMMARY.md` only |
| **`RESULTS_SUMMARY.md`** (`results/simple_to_complex/`) | Not created yet. It will be the single cross-task summary, with the "simple → complex ladder" figures `figures/04_*.png` | The file appears; then re-check every number against it |
| **Pre-merge review** | It may fix bugs, which means notebooks are re-run and tables can shift | The review commit in `git log` |
| **File locations** | When merged into `main`, paths stay the same; the plan, status and handoff files move to `Archive/` | After the merge, read this file from `Archive/2026-10-09_before_simple_to_complex/` |
| **Classification numbers** (`classification/*`) | Expected to be stable, since every run is deterministic. They are re-run only for the ROC figure and the LightGBM clean-up | Spot-check after the review commit |

## Section → sources
Paths are relative to the repo root. Tables come as `.csv`, `.html` and `.tex`; prefer the `.html` or `.csv` for reading.

**1–2. Business context and data**
- `results/simple_to_complex/data/01_raw_tables.csv`, `01_cleaning_steps.csv` and `01_population_funnel.csv`.
- Notebook `01_data_features_multicollinearity.ipynb`, sections 1 and 3.

**3. Feature engineering, multicollinearity and transformations**
- `MULTICOLLINEARITY_NOTES.md` is the main source. It covers the 41 candidates, the funnel to 24 raw features and 53 encoded columns, the three decisions, the encoding and the linear transformations, and it ends with a report-ready paragraph.
- Figures: `figures/01_skewness_before_after.png` and `01_cyclic_calendar.png`.
- Tables: `data/01_skewness.csv`, `data/linear_design_vif.csv`, and `results/feature_selection/*.csv`.
- Mixed route merged for the linear models: `data/01_route_audit_*.csv`.

**4. Evaluation design**
- Figures: `figures/01_validation_timeline.png` (the key diagram) and `01_cv_folds_classification.png` / `01_cv_folds_regression.png`.
- Label buffer and waiting period: `figures/01_label_buffer_audit.png`, `01_waiting_period_audit.png`, `data/01_buffer_audit.csv` and `01_waiting_period_audit.csv`.
- Counts: `data/01_split_summary.csv` and `01_cohort_counts.csv`.

**5. Metrics and success criteria**
- `classification/classification_success_criteria.*` and `regression/regression_success_criteria.*`.

**6–7. Stage 1: linear models**
- Transform CV: `{task}/{task}_transform_cv.*`.
- Variable selection:
  - stepwise path: `{task}_stepwise_path.*` and `figures/0{2,3}_stepwise_path.png`;
  - overlap of CV-stepwise, BIC, AIC and Lasso: `{task}_selection_overlap.*`;
  - curvature check: `figures/0{2,3}_curvature_partial_residuals.png`.
- Classification only:
  - class weights: `classification_class_weight_sensitivity.*`;
  - **class-imbalance experiment** (random under/oversampling and SMOTENC): `classification_resampling.*`, `figures/02_resampling_cv_metrics.png` and `02_resampling_calibration.png`. Conclusion: no option improves CV AP, and all of them badly worsen calibration, so none is used. The citations are marked "verify before citing".

**8. Stage 2: Random Forest**
- `{task}_rf_search_summary.*` (two-stage search on the same CV folds).
- Sensitivity: `figures/0{2,3}_rf_sensitivity.png`.
- **Tuned vs default:** `{task}_rf_tuning_gain.*`.

**9. Results and drift**
- Main tables: `{task}_model_comparison.*` and `{task}_overfitting_gaps.*`.
- Classification:
  - June deciles, capture and lift: `classification_june_deciles.*` and `figures/02_june_decile_lift.png`;
  - PR curves: `figures/02_pr_curves.png` (a ROC-curve figure, `02_roc_curves.png`, is still to come);
  - calibration: `figures/02_calibration.png`.
- Regression:
  - error by late vs on-time orders: `regression_error_by_outcome.*`;
  - derived late flag (dual framing): `regression_late_flag.*`;
  - `figures/03_predicted_vs_actual.png`.
- Drift: `{task}_drift.*`, `figures/02_drift_ap.png`, `02_drift_lift.png`, `03_drift_rmse.png` and `03_drift_mae.png`.

**10. Interpretation**
- Classification:
  - odds ratios: `classification_odds_ratios.*` and `figures/02_c2_odds_ratios.png`;
  - `seller_state_count` is flagged "not identified" (quasi-separation).
- Regression:
  - `regression_coefficients.*` (OLS with HC3) and `figures/03_r2_effects.png`;
  - diagnostics: `figures/03_ols_diagnostics.png` and `regression_ols_heteroscedasticity.*`.
- Hypothesis check: `{task}_hypothesis_check.*`.
- RF:
  - permutation importance: `figures/0{2,3}_rf_importance_{validation,june}.png`;
  - partial dependence: `0{2,3}_rf_partial_dependence.png`;
  - agreement with the linear models: `{task}_rf_agreement.*`.

**11–12. Insights and limitations**
- Run metadata: `{task}/run_metadata.json`.
- The "Notes for the report writer" in `REBUILD_STATUS.md`.
- Section 10 of `PLAN_SIMPLE_TO_COMPLEX.md`: section order, RF "variable selection" wording, and references to verify.

## Points to get right
- **`promised_lead_days`** = estimated delivery date shown at checkout − approval date. It is strong partly by construction:
  - the regression target is (actual lead days) − (promised lead days);
  - "late" means delivered after the promise.
  Interpret its effect as the *slack in the promise*, not a cause of delay.
- **June has only 137 late orders**, a 2.2% late rate vs 8.8% in Train, so June classification metrics are noisy.
- **RF vs logistic:**
  - The RF is much better in CV (AP 0.27 vs 0.14–0.15) but not on Validation or June.
  - It fails the "≥ 1.10 × logistic top-10% precision" criterion.
  - It drifts worst by August.
  Present this honestly: the more complex model is not justified out of time.
- **C2** keeps only 6 of the 24 features (1-SE rule) and performs about as well as C1 out of time.
- The old pipeline headline (LightGBM June AP 15.09%, MAE 4.85 days) belongs only in a short "what changed" note.
