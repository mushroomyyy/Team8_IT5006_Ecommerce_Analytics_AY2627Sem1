# Late-Delivery Classification: Implementation Plan

## Purpose and status

### Multi-metric evaluation

Implemented `classification_metric_report.py`, using saved prospective decisions
and probabilities without refitting or retuning thresholds. See
`EVALUATION_METRIC_GUIDE.md` and `CLASSIFICATION_METRIC_COMPARISON.md` for ranking,
decision, probability and capacity metrics. Keep mean AP as the declared primary
ranking measure; do not choose a winner by whichever secondary metric happens
to favour it. Regression MAE/RMSE await a separately agreed target. Existing
AP p-values must not be reused for secondary-metric significance claims.

### Fixed ensembles and validity gate

Implemented `classification_ensembles.py`: equal logistic/CatBoost voting and equal
four-model voting using purchase-distance OOF predictions, with strict pairing,
no validation-fitted weights and forward mature-label thresholds. See
`ENSEMBLE_AND_VALIDITY_FINDINGS.md`. Nine tests pass and fold alignment was checked
against the rebuilt dataset. The two-model blend's AP gain is not statistically
established. No learned stacking or independent final evaluation has been done.
Do not claim categorical leakage-proofness: static snapshot availability and
retrospective delivered-population selection remain explicit unresolved limits.

### Purchase timing and geographic distance follow-up

`classification_time_distance.py` adds two sequential feature bundles to the
previous enriched configuration for all four models. First add purchase weekday,
hour, month, weekend and sine/cosine encodings (10 columns). Then add mean/max
unique-seller distance, missing-distance fraction, log mean distance and distance
per positive promised lead day (5 columns). Keep the same folds and settings;
write separate outputs without overwriting prior experiments. This is sequential
ablation, not a full factorial test: distance-only is not evaluated.

Use median latitude/longitude per postcode prefix from the static geographic
reference after a broad Brazil plausibility filter. Haversine distance is a
straight-line geographic proxy, not shipping distance. Unknown coordinates stay
missing for fold-fitted imputation/native missing handling. Do not silently treat
them as zero distance. This fixed lookup contains no targets, but its historical
availability cannot be verified because it has no revision timestamps. Continue
to disclose that assumption and the eventual-delivery population limitation.

Measured follow-up findings belong in `PURCHASE_DISTANCE_FINDINGS.md`. Do not
reuse p-values from the earlier feature experiment for these new variants.

### Feature and CatBoost experiment (21 September 2026)

Implemented in `classification_experiments.py` with results interpreted in
`CLASSIFICATION_EXPERIMENT_FINDINGS.md`. Compare original and enriched feature
sets for logistic regression, Random Forest, XGBoost and CatBoost on identical
six-month development cohorts. Keep the original 180-day moving window, 30-day
gap and label-maturity gates. Do not score May-June again for this experiment.

Enrichment adds product/payment categories, state-pair routes, purchase-to-approval
duration, promise weekday, parcel ratios, a weight-missing flag, a multi-seller
flag and cyclic time encodings. No target-history or outcome features are added.
Use fixed pilot hyperparameters; CatBoost uses native categorical processing,
300 iterations, depth 6, learning rate 0.05, balanced weights and seed 42.
No validation-fold early stopping. Fit all learned transformations within training.

`classification_significance.py` implements conditional paired inference for
equal-month mean AP with 1,999 bootstrap replicates, month-stratified circular
day blocks of 7 days and sensitivity checks at 3 and 14 days. Approximate p-values
use null-centered bootstrap differences. Holm correction covers all seven
comparisons against base logistic plus the three additional within-family
feature ablations. Marginal intervals are not simultaneous confidence intervals.
This exploratory procedure does not account for retraining variability or the
adaptive search over models/features. Coverage/F1/loss inference and DM remain
planned, not implemented by this AP analysis. Use at least 2,000 replicates for
the eventual final protocol, alongside dependence diagnostics and a fresh cohort.

### Corrected pilot protocol (21 September 2026)

This update supersedes conflicting pilot assumptions below. Run the verified
classification path with `.venv/bin/python phase2/classification_backtest.py`.
Add `--include-exploratory` to reproduce the already-inspected May-June period.
No final holdout is evaluated by this command. The old `pilot_model_results.json`
is a historical snapshot; current results belong in `classification_backtest_results.json`.

- Train on a moving 180-day approval window ending 30 days before each monthly
  origin. Require every included order's promised calendar day to have ended.
  Calendar-date lateness is known at that deadline, so an overdue order is positive
  even if it has not yet arrived. Do not drop overdue pending deliveries merely
  because their delivery timestamp is after the training origin.
- The historical evaluation population remains eventual delivered orders. This is
  a retrospective population restriction, not evidence of performance on all live
  approvals. Cancellation and unresolved-order treatment remains a group decision.
- Fit all imputation/encoding/scaling inside each fold. Thresholds use only
  predictions from earlier folds with labels mature by the current origin.
  First-fold threshold is a declared 0.5 fallback. Report forward threshold metrics;
  never select a threshold on a fold and call that fold's F1 independent validation.
- Freeze exploratory thresholds at May 1 using mature development labels only.
  May-June was already inspected; do not describe it as an untouched final test.
  Select a genuinely uninspected complete period before formal model selection,
  or disclose that an independent final test is unavailable.
- Rank within each approval day. Use ceil(k * daily orders) slots. Report expected
  capture for uniform random tie-breaking at the cutoff. Lift uses random targeting
  at the same daily capacities, accounting for rounding and daily prevalence.
- Name the computed metric **average precision (AP)**, not trapezoidal PR-AUC.
- Monthly retraining and daily scoring are distinct schedules. The current pilot
  implements monthly retraining; daily retraining requires a separate simulation.
- Seven-day blocks are a starting sensitivity setting, not an established dependence
  length. Compare plausible block lengths before inferential reporting; specify
  whether the estimand weights orders or days equally. DM requires defensible loss
  differential assumptions, not just aggregation into daily losses. No p-values are
  implemented or claimed by this pilot.
- `build_order_features` accepts approved orders without delivery outcomes. Related
  tables must be snapshots available at the scoring cutoff; Olist has no revision
  history to prove that assumption. Versioned inference bundles remain future work.

Verify with `.venv/bin/python -m unittest discover -s phase2 -p 'test_*.py'`.

This document is the implementation specification for the confirmed Phase 2 classification problem. It should be sufficient for a future contributor to implement the full workflow without reconstructing decisions from chat messages or notebook history.

The corrected classification pilot has been run. `model_dev.ipynb` remains Javier's reference notebook. `model_workbench.py` supplies shared feature/model helpers and a legacy mixed-task command; use `classification_backtest.py` for the current classification workflow. The old JSON snapshot predates corrections and is not reproduced exactly by the changed helpers.

## 1. Confirmed scope

### Problem

Predict whether an order will be delivered after its promised delivery date.

### Unit of analysis

One approved order.

### Prediction time

Immediately after order approval. The production-style job is assumed to run daily for the preceding cohort of newly approved orders.

### Target

Provisional definition:

```text
late_delivery = 1 if delivered_customer_date > estimated_delivery_date
late_delivery = 0 otherwise
```

Use normalised calendar dates if the promise is date-based rather than time-based. Confirm this from the data documentation and use the same definition in feature building, training, evaluation, and reporting.

### Stakeholder and output

The primary stakeholder is the e-commerce operations/logistics team. The daily output is:

- order identifier;
- predicted late-delivery probability;
- risk rank or percentile within the daily cohort;
- threshold-based late-risk flag;
- model and feature-pipeline version;
- optional explanation fields for the selected model.

### Explicitly out of scope for this implementation

- The second required regression problem remains unresolved.
- No aggregate time-series model for daily/monthly late counts is assumed.
- No causal claim about the effect of operational interventions is made.
- No voting or stacking model is required unless individual models have first been completed and an ensemble shows stable incremental value.

## 2. Decisions still requiring group confirmation

Record the answer to each item in this file before final modelling:

1. Does "late" compare calendar dates or exact timestamps?
2. Are cancelled, unavailable, or otherwise undelivered orders outside the target population, or should they form a separate outcome?
3. Is the 30-day label-maturation gap operationally sufficient? Check the tail of observed delivery lead times.
4. Is the training window fixed at 180 days? Compare 90, 180, 270, and all-available history during validation if compute permits.
5. What daily intervention capacity should determine coverage cut-offs and threshold selection?
6. What minimum precision or maximum false-alert volume is operationally acceptable?
7. Does the instructor count Random Forest and XGBoost as one broad tree-based family or as distinct bagging and boosting families?
8. Which latest dates are complete enough to form the untouched final test period without right-censoring?

## 3. Relationship with Javier's notebook

Retain `model_dev.ipynb` as the exploratory end-to-end reference. Reuse validated logic, but move final reusable code into Python modules.

Current notebook components to retain conceptually:

- data loading and table joins;
- order-item and payment aggregation;
- binary late-delivery target;
- Random Forest and XGBoost starting configurations;
- coverage/gains concept;
- daily inference framing;
- `LOOKBACK = 30` and `PXD = 180` as hypotheses to validate.

Components to replace or extend:

- replace the final random split with a deployment-oriented moving-window backtest;
- add dummy, logistic-regression, and single-tree comparisons;
- fit all learned preprocessing inside each training fold;
- add tuning and probability calibration checks;
- add statistical uncertainty for model differences;
- separate training-time and inference-time feature construction;
- add automated schema, leakage, and reproducibility checks.

The notebook may later become a thin narrative wrapper that imports the tested modules and displays their outputs.

## 4. Proposed code structure

```text
phase2/
  classification/
    __init__.py
    config.py
    data_access.py
    feature_builder.py
    schema.py
    splits.py
    models.py
    metrics.py
    statistical_tests.py
    train.py
    inference.py
  tests/
    test_feature_builder.py
    test_schema.py
    test_splits.py
    test_metrics.py
    test_inference_contract.py
  model_dev.ipynb
  model_workbench.py
  pilot_model_results.json
  requirements.txt
```

Responsibilities:

- `config.py`: dates, window lengths, seed, feature list, target definition, and tuning spaces.
- `data_access.py`: load the bundled archive or configured raw-data directory without modelling logic.
- `feature_builder.py`: deterministic order-level aggregation using only prediction-time information.
- `schema.py`: required-column, data-type, uniqueness, range, and missingness validation.
- `splits.py`: moving-window folds, label-maturation rules, and final holdout selection.
- `models.py`: preprocessing pipelines and model constructors.
- `metrics.py`: classification, coverage, lift, calibration, and daily-loss calculations.
- `statistical_tests.py`: block bootstrap, daily loss comparison, and multiple-testing adjustment.
- `train.py`: cross-validation, tuning, final fitting, artifact metadata, and report tables.
- `inference.py`: load a fitted bundle, validate an eligible cohort, score it, and return the output contract.

Do not create this entire structure until the team confirms the open decisions. It is the intended target architecture.

## 5. Prediction-time data contract

### Allowed raw information

- order identifier and approval/purchase timestamps;
- estimated delivery date already shown to the customer;
- customer location known at approval;
- ordered items and product attributes known at approval;
- seller identifiers and locations known at approval;
- freight, price, and payment information known at approval.

### Prohibited predictor information

- carrier hand-off date;
- actual customer-delivery date;
- review score or review timestamps;
- any status update produced after the prediction timestamp;
- aggregates calculated using orders completed after the prediction timestamp.

### Candidate feature groups

1. Timing: approval hour, weekday, month, weekend, and peak/holiday indicators.
2. Promise: promised lead time in days.
3. Order composition: items, sellers, price, freight, freight-to-price ratio, weight, volume, and category diversity.
4. Geography: customer state, seller state, same-state indicator, and geodesic distance when geolocation is reliable.
5. Payments: total value, instalments, and payment-type composition.
6. Historical features, only if computed as of prediction time: seller late rate, seller order volume, route late rate, and category late rate with smoothing and cold-start defaults.

Historical features should be deferred until the base pipeline is correct because they pose the greatest leakage risk.

## 6. Deployment-compatible feature and preprocessing pipeline

### Feature builder

Implement a function with an explicit as-of date:

```python
build_order_features(raw_tables, as_of_timestamp) -> DataFrame
```

It must:

- output exactly one row per eligible order;
- be deterministic for the same inputs and timestamp;
- prohibit post-outcome columns from the returned feature schema;
- use only records available by `as_of_timestamp`;
- avoid fitting imputers, encoders, or scalers;
- expose data-quality diagnostics instead of silently dropping large numbers of rows.

### Learned preprocessing

Use `ColumnTransformer` inside a scikit-learn `Pipeline`:

- numerical: median imputation; scaling for linear models;
- categorical: most-frequent or explicit missing-category imputation, then `OneHotEncoder(handle_unknown="ignore")`;
- classifier as the final pipeline step.

Fit the entire pipeline separately inside every moving-window fold. Never preprocess the complete dataset before splitting.

### Model bundle

Persist one versioned bundle containing:

- fitted preprocessing and classifier pipeline;
- ordered input feature schema and types;
- target definition;
- training-window and label-gap configuration;
- chosen probability threshold;
- model hyperparameters and random seed;
- library versions;
- training date range and creation timestamp;
- validation summary and model version identifier.

The notebook must not be the only place where preprocessing logic exists.

## 7. Models and baselines

Implement in this order:

1. `DummyClassifier(strategy="prior")` - no-skill reference.
2. Logistic regression - simple trained classification baseline.
3. Constrained Decision Tree - simplest nonlinear tree model.
4. Random Forest - bagged tree ensemble.
5. XGBoost - boosted tree ensemble.

Do not describe the dummy model as the simple model family. It is a no-skill benchmark. Logistic regression is the simple trained baseline.

Initial tuning spaces:

### Logistic regression

- `C`: `[0.01, 0.1, 1, 10]`
- class weight: `[None, "balanced"]`
- penalty/solver: one compatible regularised setup initially

### Decision Tree

- maximum depth: `[3, 5, 8, None]`
- minimum leaf size: `[10, 25, 50, 100]`
- class weight: `[None, "balanced"]`

### Random Forest

- trees: `[200, 500]`
- maximum depth: `[8, 15, None]`
- minimum leaf size: `[5, 20, 50]`
- maximum features: `["sqrt", 0.5]`
- class weight: `[None, "balanced_subsample"]`

### XGBoost

- learning rate: `[0.03, 0.05, 0.1]`
- maximum depth: `[3, 5, 7]`
- minimum child weight: `[1, 5, 10]`
- row subsample: `[0.7, 0.9, 1.0]`
- column subsample: `[0.7, 0.9, 1.0]`
- L2 regularisation: `[1, 5, 10]`
- positive-class weight: `[1, training negative/positive ratio]`
- boosting rounds selected with early stopping inside validation only

Use randomised search or a deliberately small grid. Do not exhaustively multiply every option.

## 8. Moving-window cross-validation

### Primary design

- fixed training window: 180 approval days, provisional;
- label-maturation gap: 30 days, provisional;
- validation block: one calendar month during development;
- movement: advance one month per fold;
- final evaluation: daily predictions within the untouched test period.

For fold origin `T`:

```text
training approvals: [T - 30 days - 180 days, T - 30 days)
training label rule: estimated_delivery_date.normalize() + 1 day <= T
validation approvals: [T, next fold boundary)
```

The deadline rule applies to both classes. For this conditional eventual-delivery population, a pending order past the deadline is already late. Keep it in training rather than selecting on completion speed. Track the remaining cancellation and population-selection limitation explicitly.

### Why classification remains chronological

The primary split is determined by deployment, not by whether the target is categorical. The model will score future daily cohorts, and observed late prevalence changes materially across time. A temporal backtest therefore estimates the intended use more faithfully than a random mixture of dates.

### Secondary diagnostic

Run one stratified random split using the same feature pipeline and models. Label it as an IID diagnostic, not the headline result. Compare it with temporal performance to quantify the optimism introduced by random mixing.

### Final holdout rule

Select the latest complete period with sufficiently mature labels. Do not inspect it during feature selection, tuning, calibration, or threshold selection. Once formal work starts, record its boundaries in `config.py` and do not move them in response to performance.

## 9. Metric and threshold plan

### Primary selection metric

PR-AUC / average precision across moving-window validation predictions.

### Secondary metrics

- ROC-AUC;
- Brier score and log loss;
- precision, recall, and F1 at the chosen threshold;
- coverage and lift at top 5%, 10%, 20%, and 30%;
- confusion matrix counts;
- calibration intercept/slope or reliability plot;
- metrics by validation month and important segments.

### Threshold selection

Do not default to 0.5. Pre-specify one rule before final testing, such as:

- maximise recall while precision is at least the agreed minimum; or
- select the highest-risk fraction equal to daily review capacity; or
- minimise an explicitly stated weighted cost of missed late orders and false alerts.

Use only earlier out-of-fold predictions whose labels have matured by the decision origin. Evaluate each fold with its previously fixed threshold. A pooled threshold chosen using a fold's own labels must never be reported as independent performance on that fold. Final evaluation, when available, requires a threshold frozen using eligible development data only.

## 10. Statistical comparison plan

Statistical evidence complements, rather than replaces, effect sizes and operational metrics.

### Paired block bootstrap

Use this as the main uncertainty procedure:

1. retain paired order predictions from all candidate models;
2. group predictions by approval date;
3. resample consecutive date blocks with replacement; treat seven days as a candidate and check sensitivity to dependence length;
4. recompute metric differences for each bootstrap sample;
5. use at least 2,000 replicates for final reporting;
6. report the observed difference and 95% confidence interval.

Apply to:

- PR-AUC differences;
- coverage/lift differences;
- F1 differences at the fixed threshold;
- Brier or log-loss differences.

### Daily loss comparison

For each prediction date, calculate each model's mean additive loss:

- Brier score or log loss for probability predictions.

Compare the sequence of daily paired loss differences using a Diebold-Mariano-style statistic with a heteroskedasticity-and-autocorrelation-consistent long-run variance estimate. Report:

- mean daily loss difference;
- 95% confidence interval where available;
- test statistic and p-value;
- number of evaluated dates;
- chosen HAC lag rule.

Use this as a robustness test, not as the only evidence of model superiority. Do not apply DM directly to one aggregate F1 or PR-AUC value.

### Multiple comparisons

For subsequent experiments, designate logistic regression as the trained benchmark. Its role was informed by exploratory results; do not claim it was specified before all pilot inspection. If reporting multiple p-values, apply Holm's correction and show raw and adjusted values.

### Decision rule

A candidate replaces logistic regression only if:

1. it improves mean moving-window PR-AUC;
2. the improvement is reasonably stable across time;
3. its confidence interval and daily loss comparison do not suggest the result is driven by a few dates;
4. it provides meaningful operational improvement in coverage/precision;
5. the gain justifies complexity, calibration, inference, and interpretability costs.

Do not equate `p < 0.05` with practical superiority.

## 11. Interpretability plan

- Logistic regression: standardised coefficients or odds ratios with careful non-causal language.
- Decision Tree: visualise the pruned/shallow tree.
- All models: held-out permutation importance using the same metric.
- Selected tree ensemble: SHAP global summary and a small number of representative order explanations if time permits.

Check stability of important features across moving-window folds. A feature that appears important in only one month should not be presented as a reliable operational driver.

## 12. Required outputs

### Report tables

1. Data/cohort summary with dates, sample sizes, and late rates.
2. Moving-window fold definition and label-availability rule.
3. Cross-validation mean, standard deviation, and per-fold primary metrics.
4. Final holdout model comparison.
5. Coverage/lift at operational cut-offs.
6. Statistical comparison against logistic regression.
7. Selected hyperparameters and threshold.

### Report figures

1. Late rate and order volume over time.
2. PR curves for candidate models on the final holdout.
3. Calibration plot for leading models.
4. Performance by moving-window fold/date.
5. Coverage/gains curve.
6. Feature importance and, if used, SHAP summary.

### Machine-readable outputs

- fold-level predictions with order ID, approval date, actual label, probability, model, and fold;
- final holdout predictions in the same schema;
- metrics JSON;
- bootstrap/statistical comparison results;
- model bundle metadata;
- reproducible configuration snapshot.

Do not commit raw Olist CSV extracts or sensitive/local paths.

## 13. Testing and quality gates

### Feature tests

- exactly one feature row per order;
- known sample orders aggregate correctly;
- no forbidden post-outcome column appears in the feature set;
- historical features do not use records after `as_of_timestamp`;
- training and inference feature schemas are identical.

### Split tests

- no order appears in both training and validation/test;
- all training approvals are inside the fixed window;
- all training outcomes are observed before the validation origin;
- validation/test dates follow training dates;
- final holdout is never passed to tuning functions.

### Metric tests

- coverage is monotonic as the selected fraction grows;
- lift equals coverage divided by selected fraction;
- dummy average precision is consistent with prevalence up to tie behaviour;
- threshold selection uses validation predictions only;
- statistical comparisons preserve pairing by order/date.

### Reproducibility gates

- fixed random seed;
- dependency versions recorded;
- pipeline can be fitted twice with materially identical results;
- saved bundle reproduces in-memory probabilities on a test fixture;
- one command rebuilds all report tables from raw bundled data.

## 14. Implementation sequence

### Stage A - lock definitions

- resolve the eight open decisions in Section 2;
- freeze target and final holdout dates;
- agree on intervention-capacity metrics.

### Stage B - modularise data preparation

- extract Javier's validated joins and aggregations into `feature_builder.py`;
- add schema and leakage tests;
- reproduce the pilot dataset row counts.

### Stage C - moving-window evaluation

- implement the fold generator;
- fit dummy and logistic baselines first;
- persist out-of-fold predictions;
- verify per-fold prevalence and metrics.

### Stage D - tree models and tuning

- add constrained Decision Tree;
- tune Random Forest and XGBoost within the moving-window procedure;
- assess calibration and select the threshold using out-of-fold predictions.

### Stage E - uncertainty and final test

- implement paired block-bootstrap intervals;
- implement daily additive-loss comparison;
- freeze the selected pipeline and threshold;
- evaluate the final holdout exactly once.

### Stage F - deployment preparation

- serialise the complete fitted pipeline and metadata;
- implement the daily inference contract;
- verify training/inference parity;
- document monitoring and retraining assumptions.

### Stage G - report integration

- replace provisional pilot results in the draft;
- insert final tables and figures;
- justify the selected model using performance, uncertainty, operational value, and interpretability;
- state limitations and unresolved deployment assumptions.

## 15. Current pilot evidence, not final results

The corrected six-fold run gives mean AP 0.1981 for logistic regression and 0.1845 for XGBoost. In exploratory May-June, XGBoost has higher pooled AP (0.1338 versus 0.1217); both capture about 25.9% of late orders at nominal 10% daily capacity. There is no statistically established winner. See CLASSIFICATION_REPORT_DRAFT.md for probability-quality limitations and the full table.

Current results are in `classification_backtest_results.json`; paired predictions are in the locally ignored `classification_backtest_predictions.csv`. `pilot_model_results.json` is retained only as an obsolete historical snapshot.
