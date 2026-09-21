# Classification Model Comparison Draft

> Historical planning draft. The corrected protocol in [CLASSIFICATION_IMPLEMENTATION_PLAN.md](CLASSIFICATION_IMPLEMENTATION_PLAN.md) and results in [CLASSIFICATION_REPORT_DRAFT.md](CLASSIFICATION_REPORT_DRAFT.md) supersede this document. Earlier pilot tables are obsolete: they used whole-period capacity, arbitrary tie-breaking and an already-inspected evaluation cohort. Do not reuse their figures or treat May-June as an untouched test. The second problem remains unresolved.

## Current confirmed problem

**Prediction question:** Using information available when an order is approved, will the order be delivered after its promised delivery date?

**Target:** `late_delivery = 1` when the customer-delivery date is later than the estimated delivery date; otherwise `0`.

**Stakeholder:** E-commerce operations or logistics team.

**Operational use:** Run the classifier daily for newly approved orders, rank orders by predicted late-delivery probability, and prioritise the highest-risk orders within the team's intervention capacity.

The second project task remains a team decision. The principal options are order-level delivery-time regression or aggregate daily late-volume forecasting. Neither should be presented as final in the report yet.

## Why these classification models should be compared

The model comparison should progress from a no-skill reference to transparent and increasingly flexible trained models. This makes it possible to show whether added complexity produces genuine future-period value.

### 1. Prior-probability dummy classifier

The dummy classifier predicts from the training prevalence and does not learn relationships between order characteristics and late delivery. It is not a substantive model candidate. Its purpose is to establish the performance achievable without predictive features.

The dummy model provides:

- a PR-AUC/average-precision reference approximately equal to the positive-class prevalence;
- a Brier-score reference for probability forecasts;
- evidence that a trained model adds information beyond the historical late rate.

Accuracy from this baseline is not meaningful evidence of success because predicting every order as on time may appear accurate when late orders are rare.

### 2. Logistic regression

Logistic regression estimates the log-odds of late delivery as an additive function of the predictors. It provides the simple trained baseline for classification.

Strengths:

- transparent coefficients and direction of association;
- efficient training and daily inference;
- probability outputs suitable for risk ranking;
- a clear test of whether nonlinear models add value;
- straightforward regularisation and class weighting.

Limitations:

- assumes additive linear relationships on the log-odds scale unless interactions or transformations are added;
- may miss complex thresholds and interactions among geography, seasonality, freight, order composition, and promised lead time;
- coefficients require careful interpretation when predictors are correlated or encoded into many indicator variables.

Important hyperparameters and decisions include the penalty, regularisation strength, class weights, feature scaling, and the final probability threshold.

### 3. Decision Tree

A Decision Tree recursively partitions the feature space into groups with different late-delivery rates. It is the simplest model in the tree-based family and provides a necessary comparison before Random Forest.

Strengths:

- captures nonlinear thresholds and feature interactions;
- easy to visualise when kept shallow;
- requires little distributional structure.

Limitations:

- high variance and sensitivity to small data changes;
- prone to overfitting when depth and leaf size are not constrained;
- probability estimates can be coarse or unstable.

Important hyperparameters include maximum depth, minimum samples per leaf, split criterion, pruning strength, and class weights.

### 4. Random Forest

Random Forest combines many decision trees trained on bootstrap samples while considering random subsets of features at each split. Averaging across trees reduces the variance of a single tree and can improve generalisation (Breiman, 2001).

Strengths:

- represents nonlinearities and interactions without manual specification;
- is generally robust to scaling and monotonic transformations;
- can accommodate mixed order, payment, timing, and geography features;
- supports class weighting and probability-based ranking.

Limitations:

- less interpretable than logistic regression or a shallow tree;
- potentially slower to train and score;
- built-in impurity importance can favour features with many possible split points;
- probabilities may require calibration.

Important hyperparameters include the number of trees, number of candidate features per split, tree depth, minimum leaf size, row sampling, and class weights.

### 5. XGBoost

XGBoost constructs a sequence of regularised decision trees, with each new tree reducing errors left by the current ensemble. Unlike Random Forest's parallel bagging approach, boosting learns trees sequentially and explicitly optimises a differentiable loss function (Chen & Guestrin, 2016).

Strengths:

- strong ability to model nonlinear tabular relationships;
- explicit regularisation and shrinkage;
- row and feature subsampling can reduce overfitting;
- flexible handling of class imbalance and probability ranking.

Limitations:

- more sensitive to hyperparameter choices than simpler models;
- can overfit when depth, learning rate, and boosting rounds are poorly controlled;
- requires validation-based early stopping;
- is less directly interpretable and may require SHAP or permutation importance.

Important hyperparameters include the number of boosting rounds, learning rate, maximum depth, minimum child weight, row and column subsampling, L1/L2 regularisation, and positive-class weight.

## Report-ready comparison argument

The classification analysis compares a deliberately small sequence of models rather than a broad collection of algorithms. A prior-probability dummy classifier establishes performance without predictive features. Logistic regression then provides a transparent trained baseline and tests whether additive relationships are sufficient. A constrained Decision Tree serves as the simplest nonlinear tree model. Random Forest and XGBoost introduce two ensemble strategies: Random Forest reduces the variance of individual trees through bootstrap aggregation and random feature selection, whereas XGBoost sequentially adds regularised trees that focus on residual errors. The more complex models are retained only if they produce stable improvements on future order cohorts and provide sufficient operational value to justify their additional complexity.

## Primary evaluation design

Although the outcome is classification, the main evaluation should remain chronological because the intended use is daily prediction on future cohorts and late-delivery prevalence changes over time. Outcome type does not determine the split; the deployment process does.

Use a moving-window backtest:

- fixed training window: provisional 180 days;
- label-maturation gap: provisional 30 days;
- validation cohort: the next day or calendar period;
- move the training and validation windows forward through time;
- reserve the latest complete period as a final untouched test cohort.

For computationally expensive tuning, monthly validation blocks can approximate the daily process initially. Final evaluation should generate predictions by date so performance stability and daily loss differences can be analysed.

A stratified random split may be included as a secondary diagnostic answering a different question: how well would the model distinguish late orders if future observations followed the same mixture of time periods as the training sample? It should not replace the deployment-oriented chronological result.

## Metrics for model comparison

### Ranking and probability quality

- **PR-AUC / average precision:** primary ranking metric because late delivery is the minority outcome.
- **ROC-AUC:** secondary ranking metric.
- **Brier score or log loss:** probability-quality metric suitable for daily paired loss comparisons.
- **Calibration curve:** tests whether predicted probabilities correspond to observed late rates.

### Threshold and operational performance

- **Precision:** how many flagged orders are actually late.
- **Recall:** how many late orders are captured.
- **F1-score:** combined precision-recall summary at the chosen threshold.
- **Coverage at top k%:** share of actual late orders captured within an intervention capacity.
- **Lift at top k%:** coverage divided by the proportion of orders selected.
- **Confusion matrix:** operational counts of missed late orders and unnecessary interventions.

The probability threshold must be selected from validation predictions using a stated operational rule. Examples include maximising recall subject to a minimum precision or maximising coverage subject to the number of orders staff can review daily.

## How to support a claim that one model is better

A smaller error or larger PR-AUC is an observed difference; uncertainty analysis indicates whether that difference is stable enough to support a broader claim. The report should include effect sizes and uncertainty, not only a p-value.

### Recommended procedure

1. Generate paired predictions from every model for exactly the same order-date cohorts.
2. Report the absolute and relative difference in the primary metric.
3. Use a paired block bootstrap over dates, preferably resampling week-sized blocks, to preserve short-range temporal dependence.
4. Report a 95% confidence interval for each difference, such as `PR-AUC(model A) - PR-AUC(model B)` and `coverage@10%(A) - coverage@10%(B)`.
5. For additive probability losses, aggregate each model's loss by prediction date and apply a Diebold-Mariano-style test with a heteroskedasticity-and-autocorrelation-consistent variance estimate.
6. Correct for multiple testing if many pairwise comparisons are reported; preferably compare candidate models with one pre-specified benchmark rather than testing every possible pair.

### What the p-value would mean

For a daily Brier-loss comparison:

- null hypothesis: the two models have equal expected daily predictive loss;
- alternative hypothesis: their expected daily predictive losses differ;
- a small p-value provides evidence against equal expected loss under the test assumptions;
- it does not show that the improvement is operationally important.

The report must therefore present the mean loss difference, confidence interval, p-value, and stakeholder-facing change in coverage or precision together.

### Tests that answer different questions

- **Diebold-Mariano-style test:** suitable for a time sequence of paired additive losses such as daily mean log loss or Brier loss.
- **Paired block bootstrap:** flexible choice for PR-AUC, F1, coverage, and other aggregate metrics.
- **McNemar test:** compares paired hard classification errors at a fixed threshold, but ignores probability ranking and is not sufficient by itself.
- **DeLong test:** compares ROC-AUC values, but a standard implementation may not account for clustering and temporal dependence by prediction date.

For this project, paired block-bootstrap intervals plus a daily loss comparison are more informative than selecting one p-value test for every metric.

## Provisional pilot comparison

The first untuned chronological pilot produced the following final-cohort results. These are diagnostic only and will be replaced by moving-window cross-validation and tuned models.

| Model | PR-AUC | ROC-AUC | F1 | Coverage at top 10% | Lift at top 10% |
|---|---:|---:|---:|---:|---:|
| Prior baseline | 0.040 | 0.500 | 0.077 | 0.042 | 0.42 |
| Logistic regression | **0.097** | **0.710** | **0.161** | **0.310** | **3.10** |
| Decision Tree | 0.050 | 0.578 | 0.097 | 0.164 | 1.64 |
| Random Forest | 0.071 | 0.660 | 0.133 | 0.217 | 2.17 |
| XGBoost | 0.081 | 0.696 | 0.150 | 0.265 | 2.65 |

At this stage, logistic regression provides the strongest ranking performance and captures approximately 31% of late orders in the highest-risk 10% of orders. This is not yet evidence that it is the final model. The tree ensembles are untuned, temporal validation has not been completed, probability calibration has not been assessed, and planned geographic and historical features are absent. The appropriate provisional conclusion is that complexity has not yet demonstrated an advantage over the transparent baseline.

## Deployment-oriented pipeline requirements

The implementation should separate deterministic feature creation from fitted preprocessing and prediction:

1. **Feature builder:** accepts raw order, item, product, seller, customer, and payment records plus an `as_of_date`; outputs one row per eligible order using only information available by that date.
2. **Schema validator:** checks required columns, types, allowable values, uniqueness, and missingness before scoring.
3. **Scikit-learn pipeline:** imputes missing values, scales numerical variables where required, one-hot encodes categorical variables, and runs the classifier.
4. **Model bundle:** serialises the complete fitted pipeline, ordered feature schema, training dates, target definition, threshold, library versions, and model version.
5. **Daily scorer:** constructs the eligible daily cohort, loads the bundle, produces probabilities and risk ranks, and records data-quality and drift statistics.
6. **Monitoring:** tracks late prevalence, missingness, calibration, ranking performance, coverage, and segment-level errors after labels mature.

All transformations learned from data must be fitted inside training folds. Feature-building code should be shared between training and inference rather than copied between notebook cells and deployment code.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
