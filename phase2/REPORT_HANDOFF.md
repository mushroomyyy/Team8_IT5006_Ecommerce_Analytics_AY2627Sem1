# Report handoff: simple → complex rebuild

**For:** the agent updating the Word report (e.g. Claude desktop). Read local files only.
**Repo root:** `/Users/andrew_tjs/github/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/phase2/`, on branch `main`.
**Status (2026-10-09):** final. Every section can be written now.

## Ground rules
- Every number must come from the files listed below. Don't invent or round differently from the tables.
- Use these terms throughout: **Train**, **CV** (5 chronological folds inside Train), **Validation** (30-day out-of-time holdout, 19 Mar–17 Apr 2018, diagnostic only), **Test (June)** (the headline evaluation) and **Monitoring** (July and August, scored by frozen models).
- **No model is "selected" or called the "winner".** Every model is evaluated the same way, and the story runs from simple to complex.
- The models:
  - Classification: C0 no-skill; C1 logistic with all 24 features; C2 logistic with CV-stepwise features; C2B logistic with BIC-stepwise features; C3 tuned Random Forest; C3d default Random Forest.
  - Regression: R0a median baseline; R0b Olist promise baseline; R1 OLS; R2 OLS with CV-stepwise features; R2B OLS with BIC-stepwise features; R3 tuned Random Forest; R3d default Random Forest.
- The model key (code, short label, description, feature count) is the first section of `results/simple_to_complex/RESULTS_SUMMARY.md` and comes from `src/model_labels.py`; in the report write "Code · short label" (for example "C2 · Logistic (CV-stepwise)") on first mention.
- Primary metrics: **AP** for classification (write "AP, the area under the precision-recall curve" once) and **RMSE** for regression.
- Delete the old pipeline's text: XGBoost, LightGBM, CatBoost, the ensemble, "June-selected" models, the paired model comparison, and the limitation "June outcomes informed model selection".

## Status: final
All code work is finished, reviewed and merged into `main`. Nothing is expected to change.
- `results/simple_to_complex/RESULTS_SUMMARY.md` is the single source for numbers.
- The pre-merge review found no correctness or leakage issues.
- The plan and engineering-status files are archived in `Archive/2026-10-09_before_simple_to_complex/`. This handoff stays here for the report work.
- **Citations** (van den Goorbergh et al. 2022; Chawla et al. 2002; Strobl et al. 2007; Breiman 2001; Harrell; Tibshirani 1996; Hastie et al. 2009) appear in notebooks 02 and 03 without a "verify" marker. Check the volume and page details before putting them in the report.

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
- Transformations are **pre-specified, not tested**: log1p on the 8 skewed inputs, sine/cosine for month and weekday, the Mixed route merged into interstate, and no squared terms. The justifications are in `RESULTS_SUMMARY.md` §2.2/§3.2 and `MULTICOLLINEARITY_NOTES.md`. Variable selection is the only outcome-driven step, and it runs on the transformed features.
- Variable selection:
  - stepwise path: `{task}_stepwise_path.*` and `figures/0{2,3}_stepwise_path.png`;
  - overlap of CV-stepwise, BIC, AIC and Lasso: `{task}_selection_overlap.*`;
  - curvature diagnostic (diagnostic only; shows no strong curvature remains): `figures/0{2,3}_curvature_partial_residuals.png`.
- Classification only:
  - class weights: `classification_class_weight_sensitivity.*`;
  - **class-imbalance experiment** (random under/oversampling and SMOTENC): `classification_resampling.*`, `figures/02_resampling_cv_metrics.png` and `02_resampling_calibration.png`. Conclusion: no option improves CV AP, and all of them badly worsen calibration, so none is used. Check citation details before using them.

**8. Stage 2: Random Forest**
- `{task}_rf_search_summary.*` (two-stage search on the same CV folds).
- Sensitivity: `figures/0{2,3}_rf_sensitivity.png`.
- **Tuned vs default:** `{task}_rf_tuning_gain.*`.

**9. Results and drift**
- Main tables: `{task}_model_comparison.*` and `{task}_overfitting_gaps.*`.
- Classification:
  - June deciles, capture and lift: `classification_june_deciles.*` and `figures/02_june_decile_lift.png`;
  - PR curves: `figures/02_pr_curves.png`; ROC curves: `figures/02_roc_curves.png`;
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
- **RF vs logistic:**
  - The RF is much better in CV (AP 0.27 vs 0.14–0.15) but not on Validation or June.
  - It fails the "≥ 1.10 × logistic top-10% precision" criterion.
  - It drifts worst by August.
  Present this honestly: the more complex model is not justified out of time.
- **C2** keeps only 3 of the 24 features (1-SE rule): `customer_state`, `promised_lead_days`, `route_type`. It performs about as well as C1 out of time.
- **R2** keeps only `promised_lead_days` (1-SE rule) and is clearly worse than R1 on June (RMSE 9.37 vs 7.37). BIC (R2B, 17 variables) matches R1. Treat this as a finding about the 1-SE rule with noisy time folds.
- **CV favours the RF more than later periods do.**
  - The RF beats logistic in every CV fold (AP about 0.23–0.31 vs 0.12–0.17; see `classification/cv_fold_scores.csv`), but not on Validation and only slightly on June.
  - All five CV folds fall between September and February, which covers Black Friday and the Christmas peak. Validation, June, July and August are off-peak.
  - So the RF tuning and the stepwise selection were judged on peak-season behaviour. Present this as the reason CV over-promised for the RF; it is a key insight, not a bug.
- **1-SE stepwise is fragile here.** The large fold-to-fold differences make the 1-SE band wide, so C2 and R2 are very small. BIC (C2B/R2B) is steadier and behaves like the full model.
- **"Wrong-sign" hypotheses.**
  - More items, more sellers and weekend approval all *lower* late risk.
  - Multi-seller and Mixed-route orders have zero late orders in Train.
  - These look like data-recording quirks (e.g. how split orders are delivered or recorded), not logistics. Interpret them in the report rather than just listing "disagree".
- **Regression RF** does not beat OLS out of time: RMSE 8.40 vs 8.21 on Validation and 7.95 vs 7.39 on June.
- The old pipeline headline (LightGBM June AP 15.09%, MAE 4.85 days) belongs only in a short "what changed" note.
- **User preference:** don't call out June's small late-order count or late rate as a caveat. Report June metrics as they are.

## Evaluation checklist (cover fully, both tasks)
- **Evaluation design:**
  - Explain Train, CV (5 expanding chronological folds of 30 days, with a 45-day gap), Validation (19 Mar–17 Apr 2018, diagnostic only), Test (June: refit on all history before 2 June, scored day by day) and Monitoring (July and August, frozen models).
  - Figures: `01_validation_timeline.png` and `01_cv_folds_{classification,regression}.png`.
  - Fold table from `data/01_cv_folds.csv`; counts from `data/01_split_summary.csv` and `01_cohort_counts.csv`.
  - Label buffer and waiting-period checks: `01_label_buffer_audit.png` and `01_waiting_period_audit.png`.
- **Metrics:**
  - Classification: AP (primary), ROC-AUC and Brier; precision, recall and F1 at the top-10% capacity cut-off and at 0.5; top-10% capture and lift.
  - Regression: RMSE (primary), MAE, R², median AE and bias; error by late vs on-time orders; the derived late flag.
  - Success criteria: `{task}_success_criteria`.
- **Results:**
  - The full comparison table `{task}_model_comparison`, with `02_model_comparison_ap.png` and `03_model_comparison_rmse.png`.
  - The ladder `04_ladder.png`.
  - Overfitting gaps: `{task}_overfitting_gaps` and `04_overfitting_gaps.png`.
  - Tuned vs default RF: `{task}_rf_tuning_gain`.
  - June deciles and lift: `classification_june_deciles` and `02_june_decile_lift.png`.
  - PR and ROC curves: `02_pr_curves.png` and `02_roc_curves.png`.
  - Calibration: `02_calibration.png`.
  - Regression error by outcome, late flag, and `03_predicted_vs_actual.png`.
- **Monitoring and drift:**
  - `{task}_drift`, `04_drift_summary.png`, `02_drift_ap.png`, `02_drift_lift.png`, `03_drift_rmse.png` and `03_drift_mae.png`.
  - The models are frozen (identical fingerprints), so the changes reflect the data. The RF degrades most.

## CV folds and Random Forest tuning (explain explicitly)
- **The same 5 folds drive every decision:**
  - linear CV-stepwise + 1-SE;
  - the class-weight check;
  - the resampling experiment (resampling happens inside training folds only);
  - the RF search;
  - every CV column.
- **Per-fold results:** `{task}/cv_fold_scores.csv`. Show a fold × model table (C1, C2, C3 and R1, R2, R3); the RF's CV advantage sits in the September–February peak-season folds.
- **RF search, Stage A:** 60 random configurations with 200 trees each, varying depth, leaf size, split size, max features, bootstrap fraction, and (classification only) class weight and criterion.
- **RF search, Stage B:** a grid around the best, with 300 or 500 trees; near-ties go to the simpler model. Scored by AP or RMSE on the same folds.
- **RF tables and files:**
  - `{task}_rf_search_summary` and `{task}/rf_best_params.json`;
  - `rf_search_stageA.csv` and `rf_search_stageB.csv`, for an appendix;
  - the sensitivity figures `02_rf_sensitivity.png` and `03_rf_sensitivity.png`.
- **How the tuned settings are used:** the RF uses the 24 raw features. The tuned settings are refit on all history before 2 June for June, then frozen for July and August. Nothing is re-tuned on later data.
