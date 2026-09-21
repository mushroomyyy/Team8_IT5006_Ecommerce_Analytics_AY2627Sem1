# Phase 2 Report Writing Plan

> Historical planning draft. The corrected protocol in [CLASSIFICATION_IMPLEMENTATION_PLAN.md](CLASSIFICATION_IMPLEMENTATION_PLAN.md) and results in [CLASSIFICATION_REPORT_DRAFT.md](CLASSIFICATION_REPORT_DRAFT.md) supersede this document. Earlier pilot tables are obsolete: they used whole-period capacity, arbitrary tie-breaking and an already-inspected evaluation cohort. Do not reuse their figures or treat May-June as an untouched test. The second problem remains unresolved.

## What this report is

Phase 2 is a **6-8 page technical report** on problem definition, model building, and evaluation. It does not require a second standalone literature review like Phase 1. Instead, use a concise modelling literature section and place citations beside the methodological choices they justify.

The report should answer four questions:

1. What delivery problem are we solving, for whom, and at what prediction time?
2. How did we build valid classification and regression models without leakage?
3. Which model performs best under appropriate out-of-sample metrics, and why?
4. What can an operations stakeholder do with the predictions?

## Recommended report structure and page budget

### 1. Problem definition and stakeholder context - 0.5 to 0.75 page

Write now.

- Define the stakeholder as the e-commerce operations/logistics team.
- State the decision moment: immediately after an order is approved.
- Define the dual-framed delivery problem:
  - classification: whether the order will arrive after its promised date;
  - regression: the order's delivery lead time in days.
- Explain how the outputs will be used:
  - prioritise high-risk orders for intervention;
  - estimate operational workload and expected late-order volume;
  - improve fulfilment planning and customer communication.
- Define success in business terms, such as capturing a useful share of late orders within a limited daily intervention capacity and keeping delivery-date error acceptably low.

### 2. Data preparation and feature engineering - 0.75 to 1 page

Write the structure now; fill in final row counts later.

- Identify the Olist tables used and the order-level unit of analysis.
- Explain joins and order-item/payment aggregation.
- State inclusion criteria, date coverage, missing-value treatment, duplicates, and data-quality checks.
- Separate target variables from predictors.
- State that only information known at approval time is used as a predictor.
- Organise features into groups:
  - order timing and seasonality;
  - item count, value, freight, weight, and category diversity;
  - seller/customer location and distance proxies;
  - payment composition;
  - promised delivery window.
- Explain encoding, scaling where applicable, and preprocessing pipelines.

### 3. Model rationale and concise literature review - 1 to 1.25 pages

Start writing this now. Do not write a broad history of machine learning. Each paragraph should justify a model used in this project.

Suggested subsection title: **Model Selection and Theoretical Rationale**.

#### Baselines and linear family

- Introduce dummy baselines to show whether the trained models add predictive value.
- Use logistic regression for classification and linear/Ridge regression for regression as transparent benchmarks.
- Explain that these models test whether relatively simple additive relationships are sufficient before more complex nonlinear models are introduced.

#### Random Forest

- Explain that Random Forest combines bootstrapped decision trees and random feature subsets.
- Discuss its ability to represent nonlinearities and interactions with less variance than a single tree.
- Relate this to delivery risk, where geography, order size, seasonality, and freight may interact.
- Note disadvantages: weaker direct interpretability and the need to validate importance measures carefully.

Core source: Breiman (2001).

#### XGBoost

- Explain that XGBoost builds trees sequentially, with each tree correcting errors remaining from the existing ensemble.
- Mention regularisation, learning rate, tree depth, and early stopping as controls on model complexity.
- Justify it as a flexible tabular-data model capable of learning nonlinear delivery-risk patterns.
- Note that it requires careful tuning and can overfit without validation.

Core source: Chen and Guestrin (2016).

#### Model-family discipline

- Explain that the project compares simple baselines against a deliberately small set of tree-based models.
- Random Forest and XGBoost are already ensemble algorithms. Voting or stacking is optional and should only be included if it adds measurable out-of-sample value.
- Clarify with the instructor whether bagging and boosting will be counted as separate families or as members of one broad tree-based family.

### 4. Training, temporal validation, and hyperparameter tuning - 0.75 to 1 page

Start writing the rationale now; insert exact dates and folds after implementation.

- Describe the chronological train/validation/test design.
- Explain the label-maturation gap: orders must have had enough time to be delivered before their outcomes can enter training.
- Explain rolling-origin or expanding-window validation.
- State that fluctuating late rates make future-period testing important; random mixing could hide drift and overstate deployment performance.
- List the principal hyperparameters tuned for each model.
- State the tuning objective:
  - classification: average precision/PR-AUC;
  - regression: MAE.
- State that preprocessing is fitted inside each training fold.
- State that the final chronological test set is untouched until the model and threshold are fixed.

Your honours thesis provides a strong starting point for explaining temporal order, rolling-window evaluation, out-of-sample testing, regularisation, and XGBoost early stopping. Adapt the logic to order cohorts rather than copying the forecasting-specific details.

### 5. Evaluation framework - 0.75 to 1 page

Start writing this now.

Suggested subsection title: **Evaluation Metrics and Selection Criteria**.

#### Classification

- Explain the late-delivery class imbalance and why accuracy is insufficient.
- Make PR-AUC/average precision the main threshold-independent ranking metric.
- Use ROC-AUC as a secondary ranking metric.
- Define precision, recall, and F1-score at the chosen operating threshold.
- Explain the confusion matrix in terms of missed late orders and unnecessary interventions.
- Present coverage/gains at the top risk deciles and connect the cut-off to operational capacity.
- Select the classification threshold using validation data, not the test set.

Core source for PR-AUC under imbalance: Saito and Rehmsmeier (2015).

#### Regression

- Make MAE the primary metric because it expresses typical prediction error in days.
- Use RMSE to give more weight to large delivery-date errors.
- Report R-squared as contextual information rather than the sole criterion.
- Compare against a naive mean/median lead-time baseline.

#### Selection rule

Write the rule before results are known. For example: choose the model with the strongest temporal-validation performance on the primary metric, provided its later-period test performance is stable and it offers sufficient interpretability and operational value.

### 6. Results and model comparison - 1.25 to 1.75 pages

Wait for the completed modelling results.

- Provide one compact classification comparison table.
- Provide one compact regression comparison table.
- Show cross-validation mean and variability, not only one test score.
- Include final chronological test performance.
- Include a confusion matrix or PR curve for the selected classifier.
- Include observed versus predicted/error plots for the selected regressor.
- State whether complexity improved materially over the baseline.
- Justify final model selection using both performance and stakeholder requirements.

Avoid listing every experiment in the main report. Put full tuning grids and secondary results in an appendix.

### 7. Interpretability and actionable insights - 0.75 to 1 page

Write the method now; wait for model outputs before writing findings.

- Explain the use of held-out permutation importance across models.
- Use SHAP for the selected tree model if time permits.
- Distinguish global drivers from individual-order explanations.
- Translate findings into actions, such as:
  - which risk cohorts should be prioritised;
  - what share of late orders can be captured within a given intervention capacity;
  - which operational segments merit investigation;
  - how predicted lead time can support customer communication.
- Avoid causal claims; feature importance shows predictive association.

Core source for SHAP: Lundberg and Lee (2017).

### 8. Limitations and conclusion - 0.5 page

Draft the limitations now and refine them after results.

- Historical Brazilian marketplace data may not generalise to current or other e-commerce settings.
- Important operational variables, such as carrier capacity, weather, traffic, and fulfilment-centre conditions, are absent.
- Labels mature only after delivery, creating a delay before orders can be used for retraining.
- Late-delivery prevalence and relationships may drift over time.
- Observational predictive relationships are not necessarily causal.
- The proof of concept estimates risk; it does not establish that an intervention will prevent lateness.

Conclude by linking the chosen models and metrics back to the logistics decision rather than repeating every result.

## What to write first

Start in this order because none of these sections depends on final model scores:

1. Problem definition and stakeholder context.
2. Classification and regression target definitions.
3. Model-selection literature: baseline, Random Forest, and XGBoost.
4. Evaluation-metric justification.
5. Temporal validation and tuning methodology.
6. Planned feature groups and leakage controls.
7. Anticipated limitations.

Leave placeholders for:

- exact sample sizes and date boundaries;
- tuning results and selected hyperparameters;
- cross-validation and test metrics;
- feature-importance findings;
- the final selected models;
- evidence-based business recommendations.

## Suggested opening argument

The report should begin from the decision, not the algorithm:

> Late deliveries create operational pressure and weaken the customer experience, but intervention capacity is limited. This project therefore develops complementary order-level models for an e-commerce logistics team: a classifier that ranks newly approved orders by late-delivery risk and a regressor that estimates delivery lead time. The classification output supports targeted intervention, while the regression output supports fulfilment planning and customer communication. Both predictions are generated using only information available when an order is approved.

Treat this as a starting formulation to refine with the group, not as final submitted wording.

## Initial references

- Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32. https://doi.org/10.1023/A:1010933404324
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1705.07874
