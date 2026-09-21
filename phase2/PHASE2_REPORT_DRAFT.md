# Predicting E-commerce Delivery Performance

> Historical planning draft. The corrected protocol in [CLASSIFICATION_IMPLEMENTATION_PLAN.md](CLASSIFICATION_IMPLEMENTATION_PLAN.md) and results in [CLASSIFICATION_REPORT_DRAFT.md](CLASSIFICATION_REPORT_DRAFT.md) supersede this document. Earlier pilot tables are obsolete: they used whole-period capacity, arbitrary tie-breaking and an already-inspected evaluation cohort. Do not reuse their figures or treat May-June as an untouched test. The second problem remains unresolved.

## Phase 2 working draft

> **Draft status:** This document is a writing scaffold, not a submission-ready report. Text marked as provisional must be revised after the group confirms the prediction time, target definitions, features, validation folds, tuning procedure, and final results. The pilot results reported below come from `model_workbench.py` and must not be presented as final evidence.

## 1. Problem definition and stakeholder context

Late deliveries create a prioritisation problem for an e-commerce logistics team. Although operations staff may wish to monitor every order, intervention capacity is limited. The practical requirement is therefore to identify the orders most likely to miss their promised delivery dates early enough for action to be taken, while also estimating when those orders are likely to arrive.

The currently confirmed **classification task** estimates whether a newly approved order will be delivered after its promised delivery date. The binary target is equal to one when the recorded customer-delivery date occurs after the estimated delivery date and zero otherwise. This task is developed first because its target, stakeholder decision, and initial implementation are already defined.

Predictions are generated at order approval using only information that would plausibly be available at that time. The classifier provides a ranked intervention list. For a daily cohort of newly approved orders, the sum of predicted late-delivery probabilities can also provide an estimate of the expected number of late parcels.

The project brief also requires a regression task. The group has not yet decided whether this should be order-level delivery-time prediction or an aggregate daily late-volume forecast. The second task is therefore left explicitly unresolved in this draft rather than presenting a provisional idea as an agreed scope. **[TEAM DECISION: settle the regression target and its relationship to the classification problem.]**

The principal stakeholder is the marketplace operations or logistics team. A useful model should therefore satisfy more than statistical accuracy. It should identify a substantial proportion of genuinely late orders within a feasible intervention capacity, remain stable on future order cohorts, and provide explanations that can be translated into operational investigation. **[TEAM DECISION: define the assumed daily intervention capacity or acceptable precision/recall trade-off.]**

## 2. Data preparation and feature engineering

The analysis uses the public Olist e-commerce dataset. The order table supplies approval timestamps, estimated delivery dates, delivery outcomes, and order status. Customer, order-item, product, seller, and payment tables are joined to create one row per order. One-to-many tables are aggregated before joining so that orders containing multiple items, sellers, or payments are not duplicated.

The initial modelling population contains delivered orders approved from January 2017 onward with non-missing approval, estimated-delivery, and customer-delivery timestamps. **[TEAM DECISION: confirm whether cancelled and otherwise incomplete orders are outside the operational population or require separate treatment.]** Rows with impossible negative delivery durations are excluded. Remaining missing predictor values are imputed within the modelling pipeline so that preprocessing parameters are learned from training data only.

Candidate features are grouped as follows:

- order timing: approval hour, day of week, month, and weekend indicator;
- promised service level: number of days between approval and the estimated delivery date;
- order composition: item count, seller count, total price, freight value, freight-to-price ratio, product weight, volume, and category diversity;
- geography: customer state, primary seller state, number of seller states, and whether any seller shares the customer's state;
- payment information: total payment value, maximum instalments, and number of payment types.

The current pilot intentionally excludes `order_delivered_carrier_date` and `order_delivered_customer_date` from the predictors because these values are observed after approval and would leak information about the outcome. Customer reviews and review timestamps are excluded for the same reason. The customer-delivery timestamp is used only to construct the classification and regression targets.

Further feature engineering should test distance based on seller and customer geolocation, public-holiday or peak-period indicators, product-category groupings, and seller-history variables that are calculated using only orders completed before each prediction date. Historical aggregate features require particular care because computing them from the complete dataset would leak future information.

## 3. Model selection and literature

### 3.1 Baselines and linear models

The model comparison begins with naive baselines. A prior-probability classifier tests whether a trained classifier improves on the observed class prevalence, while a median regressor represents a simple constant delivery-time forecast. These benchmarks ensure that additional complexity is accepted only when it adds out-of-sample value.

Logistic regression and Ridge regression provide transparent linear-family benchmarks for the classification and regression tasks respectively. Logistic regression estimates an additive relationship between the predictors and the log-odds of late delivery. Ridge regression applies an L2 penalty to reduce coefficient instability when predictors are correlated. These models test whether relatively simple additive relationships are sufficient before nonlinear tree ensembles are introduced. Numerical predictors are standardised for the linear models, while categorical state variables are one-hot encoded.

### 3.2 Decision Trees and Random Forest

A Decision Tree recursively partitions the predictor space and can represent nonlinear relationships and interactions without requiring them to be specified in advance. Its predictions are readily visualised, but an unrestricted tree may have high variance. It therefore serves as the simple model within the tree-based family.

Random Forest extends this approach by fitting many trees to bootstrap samples and considering random subsets of features at each split. Aggregating the resulting predictions reduces variance and can improve generalisation relative to a single tree (Breiman, 2001). This flexibility is relevant because delivery performance may depend on interactions among promised lead time, geography, freight, seasonality, and order composition. Its disadvantages include greater computational cost and reduced direct interpretability.

### 3.3 XGBoost

XGBoost is a regularised gradient-tree-boosting method. Unlike Random Forest, which constructs trees largely independently and averages their outputs, boosting adds trees sequentially so that each stage focuses on errors remaining from the existing ensemble. Its objective combines predictive loss with penalties on model complexity, while the learning rate, tree depth, subsampling, column sampling, and number of boosting rounds govern the bias-variance trade-off (Chen & Guestrin, 2016).

XGBoost is included because it can model complex nonlinear patterns in tabular order data. However, it requires disciplined tuning and may overfit if model complexity is selected using the final test cohort. Early stopping and all hyperparameter decisions must therefore be confined to temporal validation folds.

Random Forest and XGBoost are themselves ensemble algorithms. A separate voting or stacking model is not automatically necessary. It should be included only if combining the selected models provides a reproducible improvement over their individual future-period performance.

## 4. Training and validation methodology

### 4.1 Prediction timing and label availability

The proposed deployment process runs once per day and scores orders approved in the preceding cohort. A historical order can enter model training only after its delivery outcome is known. Accordingly, validation design must respect both approval time and label availability. Including a future delivery outcome in a model supposedly trained at an earlier date would constitute target leakage even if the predictor columns themselves were valid.

### 4.2 Chronological evaluation

The final analysis should use rolling-origin or expanding-window validation. In each fold, the model is trained using outcomes observable before the fold's prediction date and evaluated on a later order cohort. The latest complete period should remain an untouched final test set until feature engineering, hyperparameters, and classification thresholds have been selected.

Random stratification is useful as a secondary diagnostic but not as the primary estimate of deployment performance. Randomly mixing orders from different dates gives the training and test sets similar time distributions and may hide changes in delivery conditions. This concern is empirically relevant in the pilot: the late-delivery rate is 5.1% in the training cohort, 12.2% during March-April 2018 validation, and 4.0% during May-June 2018 testing. A random split would blend these regimes and obscure the model's sensitivity to changing prevalence.

### 4.3 Hyperparameter tuning

Hyperparameters should be selected inside the chronological validation procedure. For classification, average precision or PR-AUC is the proposed primary search objective. For regression, mean absolute error is the proposed primary objective because it is measured directly in delivery days. Candidate parameters include tree depth, minimum leaf size, number of trees, learning rate, row and column subsampling, class weighting, and regularisation strength. Search spaces and random seeds must be reported for reproducibility. **[RESULT PENDING: insert the agreed temporal folds, search method, search spaces, and selected parameters.]**

## 5. Evaluation framework

### 5.1 Classification

Late delivery is an imbalanced outcome, so accuracy alone can reward a model that predominantly predicts the majority on-time class. Precision measures the proportion of flagged orders that are genuinely late, while recall measures the proportion of all late orders captured by the intervention list. F1-score summarises their harmonic mean at a selected threshold.

Threshold-independent discrimination is evaluated with both ROC-AUC and precision-recall AUC. ROC-AUC measures the probability that a randomly selected late order receives a higher score than a randomly selected on-time order. Precision-recall analysis places greater emphasis on performance for the positive class and is particularly informative under imbalance (Saito & Rehmsmeier, 2015). Average precision is therefore used as the primary ranking metric, with ROC-AUC as a secondary measure.

The report will also present a coverage or cumulative-gains table. Orders are ranked from highest to lowest predicted risk, and coverage at a given operational capacity is the proportion of all late orders contained in the selected highest-risk fraction. Lift divides this coverage by the fraction selected and therefore compares targeted intervention with random selection. The probability threshold must be selected using validation data and an explicit operational constraint rather than adopting 0.5 automatically.

### 5.2 Regression

Mean absolute error (MAE) is the primary regression metric because it reports the typical absolute delivery-time error in days. Root mean squared error (RMSE) gives greater weight to unusually large errors and therefore highlights models that occasionally produce severe delivery-date mistakes. R-squared is reported as a supplementary measure of variance explained but is not used alone to select the model. Every trained regressor is compared with the median-delivery-time baseline.

### 5.3 Statistical comparison

The main comparison should report performance and uncertainty across chronological folds, supplemented by block-bootstrap confidence intervals over prediction dates. If the final backtest produces a sufficiently long sequence of daily loss differences, the Diebold-Mariano test may be reported as a robustness check for additive daily losses such as log loss, Brier score, MAE, or MSE. It is not the primary test for differences in aggregate PR-AUC or F1-score.

## 6. Provisional pilot results

The following exploratory run compares untuned models using orders delivered before March 2018 for training, approvals during March-April 2018 for validation, and approvals during May-June 2018 for testing. The validation period is used to select an F1-maximising classification threshold. These choices are provisional and do not replace rolling temporal cross-validation.

### 6.1 Classification pilot

| Model | Test PR-AUC | Test ROC-AUC | Test F1 | Coverage at top 10% | Lift at top 10% |
|---|---:|---:|---:|---:|---:|
| Prior baseline | 0.040 | 0.500 | 0.077 | 0.042 | 0.42 |
| Logistic regression | **0.097** | **0.710** | **0.161** | **0.310** | **3.10** |
| Decision Tree | 0.050 | 0.578 | 0.097 | 0.164 | 1.64 |
| Random Forest | 0.071 | 0.660 | 0.133 | 0.217 | 2.17 |
| XGBoost | 0.081 | 0.696 | 0.150 | 0.265 | 2.65 |

Logistic regression provides the strongest pilot ranking performance. Its top-risk 10% contains approximately 31% of late orders, equivalent to a lift of 3.10 over random selection. This result does not establish that the linear model is final: the tree models are untuned, additional geographic and historical features are not yet included, and the substantial prevalence shift affects threshold-dependent metrics. Nevertheless, the finding demonstrates why a simple baseline is necessary and why model complexity should be justified empirically.

### 6.2 Regression pilot

| Model | Test MAE (days) | Test RMSE (days) | Test R-squared |
|---|---:|---:|---:|
| Median baseline | 5.433 | 7.459 | -0.007 |
| Ridge regression | 5.069 | 6.853 | 0.150 |
| Decision Tree | 5.002 | 6.910 | 0.136 |
| Random Forest | 4.965 | 6.826 | 0.157 |
| XGBoost | **4.761** | **6.765** | **0.172** |

XGBoost records the lowest pilot MAE at approximately 4.76 days, improving on the median baseline by about 0.67 days. However, the modest R-squared values indicate that important sources of variation remain unexplained. Distance, carrier, seller-history, peak-period, and other operational variables may be necessary to improve delivery-time estimation.

### 6.3 Implications of the pilot

The pilot supports three methodological conclusions rather than a final model selection. First, late-delivery prevalence varies sharply over time, so chronological evaluation is essential. Second, the simplest trained classifier currently outperforms the more complex alternatives, showing that added complexity cannot be assumed to improve performance. Third, classification and regression may favour different algorithms, which is acceptable because the two outputs support different operational decisions.

## 7. Interpretability and stakeholder use

The final analysis should calculate held-out permutation importance for every trained model using a common scoring rule. For the selected tree model, SHAP values may additionally show the direction and magnitude of each feature's contribution to global and order-level predictions (Lundberg & Lee, 2017). Built-in impurity importance should not be the only interpretation method because it can favour variables offering many potential split points.

Interpretation must remain predictive rather than causal. For example, a high freight-to-price ratio may help identify risky orders but does not by itself establish that changing freight charges would prevent late delivery. The operational value lies in using predictive patterns to prioritise investigation and intervention.

Potential stakeholder outputs include a daily risk-ranked order list, expected late-order count obtained by summing probabilities, coverage at different staffing capacities, expected delivery dates, and segment-level monitoring by geography or product group. **[RESULT PENDING: replace these general possibilities with actions supported by final feature and error analysis.]**

## 8. Limitations

The Olist data describe one historical Brazilian marketplace and may not generalise to other platforms, countries, or contemporary delivery networks. Several operational determinants, including carrier capacity, route conditions, traffic, weather, and fulfilment-centre workload, are unavailable. The modelling sample currently focuses on delivered orders, while a live system cannot know at approval whether an order will eventually be cancelled or remain incomplete.

Delivery outcomes also mature with delay, limiting how quickly recent orders can be incorporated into training. Temporal shifts in late-delivery prevalence may weaken fixed thresholds and probability calibration. Model monitoring and periodic recalibration would therefore be required in deployment. Finally, the observational analysis identifies predictive associations rather than causal effects, and it does not demonstrate that an intervention prompted by the model would prevent lateness.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1705.07874

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
