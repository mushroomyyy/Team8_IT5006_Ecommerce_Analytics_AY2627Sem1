# Ensembles and validation audit — 21 September 2026

## Results

Fixed equal-weight soft voting was tested using the purchase-time-plus-distance
variants. One ensemble averages logistic and CatBoost probabilities; the other
averages logistic, Random Forest, XGBoost and CatBoost. No weights were learned
from validation labels. Each component was trained on the same eligible historical
window. Each ensemble's F1 threshold uses earlier, mature out-of-fold labels only,
with 0.5 as the first-fold fallback. RF and boosting are themselves ensembles;
these experiments additionally combine distinct model predictions.

| Model | Equal-month mean AP | Daily top-10% coverage | Brier score |
|---|---:|---:|---:|
| Original logistic benchmark | 0.19814 | 21.48% | 0.1554 |
| Purchase-distance logistic | 0.19777 | 22.23% | 0.2261 |
| Purchase-distance CatBoost | 0.19115 | 21.69% | 0.1579 |
| Equal logistic + CatBoost | 0.20124 | 22.47% | 0.1730 |
| Equal all four | 0.19613 | 21.86% | 0.1416 |

The two-model blend has the highest observed mean AP in this batch, but its
improvement over original logistic is not statistically established: AP difference
+0.00310, marginal 95% seven-day bootstrap interval [-0.00188, 0.00844],
Holm-adjusted approximate p=0.792. It remains nonsignificant with three- and
fourteen-day blocks (adjusted p=0.9945 and 0.642). The four-model blend also does
not establish an AP improvement (seven-day adjusted p=0.831).

The testing family includes six comparisons to original logistic: the four
purchase-distance individual models and both ensembles. Holm correction is
within this batch and separately for each block length, not a correction for all
adaptive experiments across the project. Method: 1,999 paired circular date-block
resamples within each fixed validation month, equal-month mean AP, null-centered
approximate two-sided p-values. Marginal intervals are not simultaneous intervals.
This conditions on fitted predictions and the six months; retraining variability,
new-month uncertainty and model-search bias are not incorporated.

Coverage and Brier changes are descriptive, not tested here. The two-model blend
has worse Brier score than original logistic: a ranking improvement is not evidence
of calibration. Do not deploy these class-weighted scores as calibrated late-order
probabilities or sum them into expected counts without further assessment.

## What has been checked

- Rebuilt the current modelling dataset and all six folds, then matched all 12
  timing/distance variants to the expected order IDs, labels, dates and label-maturity
  timestamps in every fold. All contain the same 40,056 development orders.
- Training/validation order sets are disjoint; training labels meet the origin
  maturity gate and the fixed historical window. Later folds may legitimately
  train on earlier validation cohorts once eligible, matching monthly deployment.
- Base/enriched feature builders do not require delivery outcomes or order status;
  future outcomes are not predictors. Purchase-time and geography code reads only
  predictor fields, with no target aggregation.
- Imputation/scaling/encoding are fit on each training fold. CatBoost receives only
  training labels when fitting native categorical transformations. No validation
  early stopping is used. Geographic coordinates are a fixed reference lookup,
  not a target-derived encoder; see the important availability assumption below.
- Threshold audits and label-perturbation tests prevent current/later validation
  labels from selecting that fold's threshold.
- Ensemble inputs must match on order, fold, approval date, label-maturity date,
  outcome and sample. Invalid probabilities, missing members, duplicate orders
  and mismatched folds are rejected. Label changes do not change voting weights
  or the arithmetic blend; model weights are fixed at equal values.
- Nine automated tests pass, including calendar extraction, geographic reference
  handling, known distances, weighted AP/ties, ensemble alignment and existing
  leakage tests. No May–June rescoring or final holdout evaluation was performed.

## What is NOT proven

1. **Historical feature availability.** Olist has no revisions for payment, item,
   seller/product details, promised dates or geographic reference data. Their values
   are assumed available at approval. Tests cannot prove that assumption. The full
   postcode lookup is static and outcome-free but not historically versioned.
2. **All-live-order validity.** Evaluation still conditions on eventual delivered
   orders and excludes negative observed delivery durations. This retrospective
   population is not identifiable in full at approval. Cancellation/unresolved-order
   treatment must be settled before claiming prospective operational validity.
3. **Independent final performance.** These months have repeatedly informed model
   and feature choices. They are development validation, not untouched test data.
   The earlier inspected May–June period must not be relabelled a final holdout.
4. **Independence across time/entities.** Shared sellers and temporal dependence
   are expected for the existing-seller deployment task. Day blocks only partially
   address uncertainty; they do not establish performance on entirely new sellers
   or future regimes. Those require different evaluation designs.

Therefore describe the work as **leakage-guarded development evaluation under
explicit data-availability/population assumptions**, not categorically leakage-proof
or final deployment validation. Passing tests does not remove the above limitations.

## Next steps

Keep original logistic as the benchmark and the fixed logistic/CatBoost blend as
an exploratory candidate. Do not optimise voting weights against these same outer
fold labels. Learned stacking needs a separate inner chronological OOF procedure,
with mature labels for the meta-model, followed by untouched outer scoring.
Before final testing, freeze the population definition, available-at-scoring
feature contract, model/ensemble settings, threshold/capacity and test protocol.
Reserve a sufficiently mature genuinely uninspected cohort if available; otherwise
disclose the absence of independent testing rather than manufacture a holdout claim.

## Reproduce

```bash
.venv/bin/python phase2/classification_ensembles.py
.venv/bin/python -m unittest discover -s phase2 -p 'test_*.py'
```

Requires the existing corrected backtest and purchase-distance predictions.
Writes `classification_ensemble_results.json`, local ignored
`classification_ensemble_predictions.csv` and `classification_ensemble_significance.json`.
