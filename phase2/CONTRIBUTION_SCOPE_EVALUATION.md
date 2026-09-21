# Phase 2 Contribution Scope: Evaluation, Model Literature, and Interpretation

> Historical planning draft. The corrected protocol in [CLASSIFICATION_IMPLEMENTATION_PLAN.md](CLASSIFICATION_IMPLEMENTATION_PLAN.md) and results in [CLASSIFICATION_REPORT_DRAFT.md](CLASSIFICATION_REPORT_DRAFT.md) supersede this document. Earlier pilot tables are obsolete: they used whole-period capacity, arbitrary tie-breaking and an already-inspected evaluation cohort. Do not reuse their figures or treat May-June as an untouched test. The second problem remains unresolved.

Prepared as a working plan for contributing to `model_dev.ipynb` and the Phase 2 report.

## Recommended problem framing

Use **one operational delivery problem with two complementary outputs**:

1. **Classification:** At the time an order is approved, will it be delivered after its promised delivery date?
   - Unit of analysis: one order.
   - Target: `late = 1` when `order_delivered_customer_date > order_estimated_delivery_date`; otherwise `0`.
   - Stakeholder output: a daily ranked list of at-risk orders for the operations/logistics team.

2. **Regression:** At the time an order is approved, how many days will delivery take?
   - Unit of analysis: one order.
   - Target: delivery lead time in days, such as `order_delivered_customer_date - order_approved_at`.
   - Stakeholder output: a predicted delivery date or expected lead time for fulfilment planning and customer communication.

This satisfies the project requirement to cover classification and regression without introducing an unrelated second business problem.

### What to do with the monthly late-parcel question

Do **not** make "how many parcels will be late this month?" a separate model at this stage. The dataset contains relatively few complete months, which gives an aggregate forecasting model little training data and introduces a third prediction setup.

Instead, derive operational volume estimates from the order-level classifier:

- expected late orders for a daily or monthly cohort = sum of predicted late probabilities;
- actual late orders captured in the top 10%, 20%, or 30% of risk scores = coverage/gains table;
- expected workload by date, seller state, customer state, or product category = aggregate the order-level risk scores.

## The most valuable contribution to make now

Own the **evaluation and model-selection workstream**, with supporting literature and report writing. This complements the teammate's current ownership of loading, joining, basic feature engineering, and initial model code.

Suggested ownership:

1. Define the classification and regression targets precisely.
2. Design the temporal train/validation/test procedure.
3. Define metric tables and threshold-selection rules.
4. Write the Random Forest, XGBoost, baseline-model, and evaluation literature.
5. Add model comparison, feature importance, and interpretability analysis.
6. Translate results into logistics actions and limitations.

## Important review of the current notebook

### 1. Replace the final random split with a temporal evaluation

The notebook currently uses a stratified proportional `train_test_split`. Stratification preserves the class ratio, but randomly mixing older and newer orders does not reproduce the proposed daily deployment setting.

Month-to-month or day-to-day fluctuation in late-delivery rates is evidence of temporal drift. It makes a chronological test **more important**, because the model must demonstrate that it generalises from earlier orders to later orders. A random split can conceal that drift and give an optimistic estimate.

Recommended structure:

- reserve the latest completed period as an untouched chronological test set;
- use rolling-origin or expanding-window validation within the earlier data for tuning;
- include a sufficient outcome-maturation gap so every training label is known by the prediction date;
- use stratification only for a secondary random-split benchmark, not as the main operational result;
- report performance by day or month as well as overall performance to show stability.

This is directly aligned with the honours thesis, which explains that time order is critical, random splits can cause leakage and over-optimistic results, and rolling-window evaluation better represents out-of-sample forecasting.

### 2. Complete both required prediction tasks

The notebook currently trains only classifiers. It creates `dpt_var_est_vs_delivered_days`, but does not train or evaluate a regression model. Decide whether that variable represents days early/late relative to the promise or replace it with delivery lead time in days. Document the choice consistently in the problem statement, code, and report.

### 3. Introduce a simple baseline

The project brief asks the team to start with the simplest model in a family and justify added complexity. Add:

- classification baseline: majority-class/dummy classifier and logistic regression;
- regression baseline: mean/median dummy regressor and linear or Ridge regression;
- tree comparison: a single decision tree before Random Forest;
- advanced tree model: XGBoost, if it adds demonstrable value.

Random Forest and XGBoost are already **ensemble learning methods**. Random Forest bags independently trained trees; XGBoost sequentially adds trees to correct preceding errors. Voting or stacking is optional and should only be added if it improves the untouched test results enough to justify the extra complexity.

### 4. Keep tuning separate from final testing

Use temporal validation folds for hyperparameter selection. Suitable search objectives are:

- classification: average precision/PR-AUC as the primary tuning score;
- regression: negative MAE as the primary tuning score;
- XGBoost: early stopping on the current validation fold, never on the final test set.

Record the search space, selected parameters, validation score, random seed, and computation budget. The final chronological test set should be evaluated only after the model and decision threshold have been selected.

## Evaluation plan

### Classification

Because late deliveries are the minority class, accuracy should not drive model selection.

Report:

- **PR-AUC / average precision:** primary threshold-independent metric for ranking late orders under class imbalance;
- **ROC-AUC:** secondary ranking metric;
- **precision:** of orders flagged as risky, how many were actually late;
- **recall:** of all late orders, how many the intervention list captured;
- **F1-score:** balance of precision and recall at the selected threshold;
- **confusion matrix:** counts of operational successes and errors;
- **coverage/gains by risk decile:** cumulative share of actual late orders captured within the highest-risk share of orders.

The current "coverage" calculation is essentially cumulative recall after orders are ranked by predicted risk. Retain it because it is stakeholder-friendly, but label it clearly and add lift relative to random targeting.

Do not accept the default probability threshold of 0.5 automatically. Select a threshold using validation data and an explicit operational trade-off, for example:

- maximise recall subject to a minimum acceptable precision; or
- maximise late-order coverage subject to the number of orders the operations team can intervene on daily.

### Regression

Report:

- **MAE:** primary metric because the typical absolute error is directly interpretable in delivery days;
- **RMSE:** secondary metric that penalises large delivery-date errors more strongly;
- **R-squared:** descriptive measure of variation explained, not the sole selection criterion;
- residual/error plots by time and relevant segments to identify systematic under- or over-prediction.

Compare each model with a naive baseline, such as the training-set median delivery lead time. If the platform's promised date is treated as an existing operational forecast, compare the proposed model against that promise where the comparison is logically equivalent.

### Final model selection

Use a single comparison table containing:

- validation mean and variability across temporal folds;
- untouched chronological test metrics;
- performance stability across time and important segments;
- inference complexity and reproducibility;
- interpretability and usefulness to the logistics stakeholder.

Select the final model based on the stakeholder objective, not merely the largest AUC or smallest RMSE.

## Feature importance and interpretation

Use held-out permutation importance as the common comparison across models. For the selected tree model, SHAP can then provide:

- global importance and direction of effects;
- order-level explanations for why an order was flagged;
- segment-level patterns useful to operations.

Avoid relying only on built-in impurity-based Random Forest importance, which can favour variables with many possible split points. Interpret associations as predictive rather than causal.

## What transfers from the honours thesis

| Phase 2 requirement | Material already demonstrated in the thesis | Adaptation needed here |
|---|---|---|
| Problem and stakeholder context | Clear motivation, forecasting objective, and user relevance | Reframe for logistics operations and order-level intervention |
| Preprocessing methodology | Large-scale text filtering, aggregation, index construction, lags, and PCA | Describe Olist joins, missingness, aggregation, and prediction-time availability |
| Feature engineering | Lags, principal components, sentiment indices | Add distance, timing/seasonality, order composition, seller/customer geography, and promise-window features |
| Model descriptions | Detailed LASSO/Elastic Net, Random Forest, blockwise RF, and XGBoost literature | Shorten and adapt from regression forecasting to classification plus regression |
| Hyperparameters | Regularisation, RF settings, XGBoost learning rate/regularisation/early stopping | Tune within temporal folds and document search spaces |
| Classification metrics | Not covered | Add precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, threshold selection, and coverage/lift |
| Regression metrics | RMSE and formal comparison with a benchmark | Add MAE and R-squared; retain RMSE and baseline comparison |
| Cross-validation | Strong discussion of temporal ordering and rolling-window evaluation | Implement rolling/expanding temporal validation for order cohorts |
| Interpretability | RF variable importance across time and forecast horizons | Use held-out permutation importance and optionally SHAP |
| Model comparison/selection | Benchmark-based out-of-sample comparison and statistical testing | Compare simple baselines with RF/XGBoost using business-aligned metrics |
| Ensemble/stacking | RF and XGBoost are ensemble methods; no voting/stacking | Stacking is optional, not a gap that must be filled |
| Actionable insights | Interpretation for policy/private-sector users | Convert risk scores into intervention capacity, segments, and expected late-order volume |
| Limitations | Data, computation, model scope, and generalisability limitations | Add historical-data age, geography, label delay, unobserved logistics factors, drift, and non-causal interpretation |

## Suggested literature starter set

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5-32. https://doi.org/10.1023/A:1010933404324
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD 2016, 785-794. https://doi.org/10.1145/2939672.2939785
- Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems 30. https://arxiv.org/abs/1705.07874
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

The Random Forest and XGBoost explanations and several of these sources already appear in the honours thesis, so they can be adapted efficiently while avoiding verbatim reuse where the task and model objective differ.

## Concrete first deliverable

Produce a short evaluation subsection and companion notebook functions that:

1. create chronological training, validation, and test cohorts with a label-maturation gap;
2. calculate the full classification and regression metric tables;
3. calculate coverage and lift at operational capacity cut-offs;
4. compare baseline, Random Forest, and XGBoost results;
5. generate temporal stability and feature-importance plots;
6. state the final selection rule before viewing final test results.

This is a distinct contribution, closes several current notebook gaps, and reuses the strongest parts of the honours thesis without duplicating the teammate's data-pipeline work.
