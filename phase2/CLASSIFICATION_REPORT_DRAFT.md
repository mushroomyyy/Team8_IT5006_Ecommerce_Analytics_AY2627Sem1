# Late-Delivery Classification

## Working report draft

Metric definitions and the expanded results are now available in
[the evaluation guide](EVALUATION_METRIC_GUIDE.md) and
[the metric comparison](CLASSIFICATION_METRIC_COMPARISON.md). Average precision
is the primary ranking metric, not precision at a single threshold. The report
also covers explicitly labelled trapezoidal PR-AUC, ROC-AUC, prospective precision/
recall/F1, balanced accuracy, MCC, confusion counts, Brier/log loss and matched
daily-capacity precision/coverage/lift. MSE on binary probabilities is the Brier
score; regression MAE/RMSE in days or parcel counts belong to a separately defined
regression target. Secondary metric differences are descriptive, not established
by the existing AP significance tests.

Latest follow-up: purchase-time/distance features and fixed equal-weight voting
are now evaluated; see [the validation audit and ensemble findings](ENSEMBLE_AND_VALIDITY_FINDINGS.md).
The logistic/CatBoost blend reaches mean AP 0.20124 versus original logistic's
0.19814, but the gain is not statistically established (seven-day paired bootstrap
95% marginal interval for the difference [-0.00188, 0.00844], Holm-adjusted
approximate p=0.792). Daily top-10% coverage is 22.47% versus 21.48%, descriptively.
This is development evidence, not independent testing; point-in-time snapshot
availability and the eventual-delivery population restriction remain unresolved.

> Updated 21 September 2026 after the corrected pilot and feature/CatBoost experiment. Six moving-window folds and exploratory paired AP comparisons have been run. Tuning, calibration and independent final evaluation remain pending. See [the experiment findings](CLASSIFICATION_EXPERIMENT_FINDINGS.md) for the latest ablation and uncertainty results. The second required regression problem remains unresolved.

## 1. Problem definition and stakeholder context

Late deliveries create both an operational prioritisation problem and a customer-experience risk for an e-commerce marketplace. Although a logistics team may wish to monitor every order, intervention resources are finite. The practical task is therefore to identify the newly approved orders most likely to miss their promised delivery dates early enough for investigation or intervention.

This study develops an order-level binary classifier that estimates late-delivery risk when an order is approved. An order is provisionally labelled late when its recorded customer-delivery date falls after its estimated delivery date. Predictions are generated using only information plausibly available at approval. The resulting probability is used to rank each daily cohort from highest to lowest risk.

The primary stakeholder is the marketplace operations or logistics team. The classifier is intended to support a daily risk-ranked intervention list rather than replace operational judgement. Model success is therefore defined in two ways. First, it should discriminate late from on-time orders on future chronological cohorts. Second, it should capture a useful share of actual late orders within the number of orders staff can review. **[TEAM INPUT REQUIRED: specify the assumed daily intervention capacity or minimum precision requirement.]**

## 2. Data preparation and prediction-time features

The analysis uses the public Olist e-commerce dataset. Orders are joined with customer, item, product, seller, and payment information to create one modelling row per order. One-to-many item and payment records are aggregated before joining so that multi-item and multi-payment orders are not duplicated.

The initial population contains delivered orders with non-missing approval, estimated-delivery, and customer-delivery timestamps. **[TEAM INPUT REQUIRED: confirm whether cancelled or otherwise incomplete orders are outside the operational population or require separate treatment.]** Orders with impossible negative delivery durations are excluded. Remaining missing predictor values are imputed within the fitted modelling pipeline.

Candidate predictors include approval timing, the promised delivery window, item and seller counts, product value, freight, product weight and volume, category diversity, seller/customer geography, same-state delivery, and payment composition. Carrier hand-off dates, customer-delivery dates, review outcomes, and review timestamps are excluded from the predictors because they are not known at approval and would leak post-outcome information.

The implementation separates outcome-independent feature creation from historical target construction. The feature builder accepts approved orders without delivery outcomes; fitted pipelines learn imputation, scaling and encoding within each training window. Callers must supply related-table snapshots available at the scoring cutoff. Olist does not contain the revision history needed to verify this assumption retrospectively. Serialization, production schema checks and deployment monitoring remain planned work.

## 3. Model comparison and literature

The comparison progresses from a no-skill reference to transparent and increasingly flexible trained models. This structure tests whether added complexity produces stable future-period value rather than assuming a complex model will perform better.

A prior-probability dummy classifier establishes performance without predictive features. Logistic regression is the simple trained baseline. It models late-delivery log-odds as an additive function of the predictors, is computationally efficient, and provides relatively transparent coefficients and probability estimates. Its main limitation is that nonlinearities and interactions must be specified or transformed explicitly.

A constrained Decision Tree is used as the simplest nonlinear tree model. It recursively divides orders into groups with different late rates and can represent thresholds and interactions, but an unrestricted tree is unstable and prone to overfitting.

Random Forest reduces the variance of a single tree by averaging predictions from many trees trained on bootstrap samples and random subsets of features (Breiman, 2001). It can capture interactions among order composition, geography, seasonality, freight, and promised delivery time without requiring those relationships to be specified in advance. Its disadvantages include reduced direct interpretability, computational cost, and potentially uncalibrated probability estimates.

XGBoost instead adds regularised trees sequentially, with each new tree focusing on errors remaining from the current ensemble. Learning rate, depth, sampling, regularisation, and boosting rounds control model complexity (Chen & Guestrin, 2016). It is a flexible candidate for structured order data but requires careful tuning and validation-based early stopping.

Random Forest and XGBoost are already ensemble learning methods. A separate voting or stacking model will be considered only if the individual models have been fully evaluated and their combination provides a stable improvement that justifies added complexity.

## 4. Moving-window validation

The primary evaluation is chronological because the intended deployment scores future daily order cohorts and late-delivery prevalence changes over time. Classification does not itself require random splitting; the correct split depends on the deployment process.

The corrected backtest uses a fixed 180-day approval window ending 30 days before each monthly origin. Training eligibility additionally requires that the promised calendar day has ended. In the eventual-delivery population, an order still pending after that deadline is already known to be late; its eventual delivery timestamp need not be observed for classification. Applying the deadline rule to both classes retains overdue pending orders and avoids selecting only quickly completed deliveries. Six monthly folds cover November 2017 through April 2018. Models refit monthly and daily cohorts are scored with the current model. This is not a simulation of daily retraining.

May-June 2018 was already inspected in the earlier pilot and is explicitly exploratory. It is evaluated only with an opt-in command. No independent final holdout has been evaluated; a sufficiently complete, uninspected cohort must be reserved before formal model selection, or the absence of independent testing must be disclosed.

The 180-day window and 30-day gap are hypotheses rather than fixed facts. The gap will be compared with the empirical delivery-duration distribution, and alternative window lengths will be evaluated if resources permit. A stratified random split may be reported as a secondary IID diagnostic, but not as the primary deployment estimate.

## 5. Metrics and threshold selection

Late delivery is an imbalanced target, so accuracy is not used as the main selection measure. A model that labels nearly every order as on time may achieve high accuracy while failing to identify actionable risk.

Average precision (AP) is the primary ranking metric. It summarises the precision-recall curve using recall increments and differs from trapezoidal PR-AUC. Precision-recall analysis concentrates on the positive class (Saito & Rehmsmeier, 2015). ROC-AUC is a secondary discrimination measure. Report fold prevalence alongside AP because AP changes with prevalence.

Probability quality is evaluated with Brier score and log loss; reliability plots and calibration fitting remain pending. Coverage selects ceil(k times daily orders) orders within each day and aggregates captured positives over all days. At a tied cutoff, the result is expected capture under uniform random tie-breaking. Lift compares captured positives with expected random capture at exactly the same daily capacities. The selected fraction is reported because rounding slightly changes the nominal budget.

For each fold, the pilot maximises F1 using only earlier out-of-fold predictions whose promised days have ended by the fold origin. The first fold uses a declared 0.5 fallback. F1 is then evaluated prospectively on the new fold. The exploratory May-June threshold is frozen using development labels available by May 1, never labels arriving later. F1 maximisation remains provisional until the stakeholder specifies capacity or error costs.

## 6. Statistical model comparison

A larger AP or lower Brier score is an observed difference, not by itself evidence of superiority. The follow-up experiment estimates differences in equal-month mean AP using paired circular date-block resampling within each development month. Seven-day blocks are the primary specification, with three- and fourteen-day sensitivity runs, each using 1,999 replicates.

Paired predictions for the same orders and dates have been saved. Approximate two-sided p-values use null-centered bootstrap differences and Holm adjustment across ten comparisons. Intervals are marginal percentile intervals. These conditional development analyses do not include training variability, new-month variability or adaptive model-search uncertainty. Coverage, F1 and additive-loss inference remain planned. Fold standard deviations below describe variation across months; they are not confidence intervals or standard errors.

A supplementary Diebold-Mariano comparison of daily Brier or log losses may be appropriate if the loss differential has defensible stability and dependence properties. Daily aggregation alone does not establish validity. DM has not been run. Logistic regression is the benchmark for the follow-up comparisons; it was retained after the initial pilot, so it should not be described as pre-specified before all exploratory analysis.

Model superiority will be claimed only when the candidate demonstrates a stable future-period improvement, an uncertainty interval consistent with improvement, meaningful operational gains in coverage or precision, and sufficient benefit to justify its complexity.

## 7. Provisional pilot findings

The corrected run contains 40,056 development predictions and 13,034 exploratory predictions per model. Paired predictions cover identical orders across all five models. Features and model configurations were fixed for this run; no hyperparameter search was performed. These results supersede the old single-split tables.

| Model | Six-fold AP mean ± SD | Exploratory AP | Exploratory ROC-AUC | Daily top-10% coverage | Daily lift |
|---|---:|---:|---:|---:|---:|
| Prior baseline | 0.1053 ± 0.0554 | 0.0620 | 0.6731 | 10.21% | 1.00 |
| Logistic regression | **0.1981 ± 0.1024** | 0.1217 | 0.7710 | **25.90%** | **2.54** |
| Decision Tree | 0.1551 ± 0.0758 | 0.0937 | 0.7280 | 17.47% | 1.71 |
| Random Forest | 0.1708 ± 0.0924 | 0.1215 | 0.7523 | 24.00% | 2.35 |
| XGBoost | 0.1845 ± 0.0912 | **0.1338** | **0.7827** | **25.90%** | **2.54** |

Logistic regression leads mean validation AP, while XGBoost leads pooled exploratory AP. Neither is a statistically established winner. The monthly-refitted dummy assigns a different constant score in May and June, so pooled AP and ROC-AUC can exceed chance by ranking months; it cannot rank orders within a day and its daily lift is exactly 1.0. This illustrates why pooled ranking metrics and daily intervention metrics answer different questions.

Exploratory Brier scores are 0.0432 for the prior baseline, 0.1920 for logistic regression, 0.2221 for Decision Tree, 0.1051 for Random Forest and 0.1612 for XGBoost. All trained models in this run use imbalance weights. Their probabilities need calibration assessment and comparison with unweighted variants before use as expected-count estimates. Good ranking does not imply reliable probability values.

Monthly development late rates range from 4.62% to 19.20%. The large fold standard deviations show substantial temporal variability. Changes in prevalence may reflect seasonality, sampling and other shifts; prevalence variation alone does not prove a specific form of concept drift.

## 8. Interpretability and operational output

### Follow-up feature and CatBoost evidence

Nineteen approval-time predictors were added, including product/payment categories,
state routes, parcel ratios and cyclic timing features. Under the same folds,
enrichment increased mean AP for Random Forest from 0.1708 to 0.1850 and CatBoost
from 0.1830 to 0.1910 (seven-day Holm-adjusted approximate p=0.0050 and 0.0075,
respectively). Both gains persisted across block-length sensitivity checks.
Enriched XGBoost reached 0.1876; enriched logistic reached 0.1975, neither a
statistically supported feature gain. Original logistic remained highest at 0.1981.
Its advantage over enriched CatBoost was sensitive to block length (adjusted
p=0.069, 0.0195 and 0.012 for 3, 7 and 14 days). No final winner is declared.
These experiments did not rescore May–June. See the linked findings for full
effect sizes, intervals, operational coverage and limitations.

CatBoost processes categorical predictors within training rather than requiring
external one-hot encoding. Here, it receives native category columns and uses fixed
boosting rounds with no scoring-fold early stopping; see the
[CatBoost documentation](https://catboost.ai/docs/en/features/categorical-features).
Outcome-removal tests pass, but related-table snapshot availability and the
eventual-delivery population restriction remain unresolved limitations.

### Planned interpretation

The final analysis will compare held-out permutation importance across models using a common scoring metric. Logistic-regression coefficients can provide a transparent additive description, while SHAP may be used for the selected tree ensemble to describe global and order-level contributions. All interpretations will be presented as predictive associations rather than causal effects.

The proposed daily output contains the order ID, risk score, cohort rank, threshold flag and model version. Summed probabilities should be used as expected late-order counts only after probability calibration has been validated for the intended population. Coverage by staffing capacity and segment monitoring are other candidate outputs.

## 9. Limitations to refine after implementation

- The data represent one historical Brazilian marketplace and may not generalise to other settings.
- Carrier capacity, route conditions, weather, traffic, and fulfilment-centre workload are unavailable.
- Restricting the sample to delivered orders may create a gap between training and the population observed at approval.
- Outcomes mature with delay, limiting how quickly recent data can be used for retraining.
- Temporal drift may weaken fixed thresholds and probability calibration.
- Predictive feature importance does not establish causal intervention effects.
- The pilot uses untuned configurations, six monthly folds and a previously inspected exploratory period; it does not provide independent final-test evidence.
- Outcome timing is enforced in features, training and threshold selection, but eventual-delivery population selection and unversioned raw table snapshots remain limitations.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
