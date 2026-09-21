# Purchase timing and geographic distance

## Experiment design

This follow-up distinguishes order placement from approval. The existing enriched
features already contained approval weekday/hour/month and cyclic encodings, but
not purchase weekday/hour/month. Approval can occur later than purchase.

The comparison keeps the same six development folds, monthly refit cadence,
180-day moving training window, 30-day gap and label-maturity checks. It adds:

1. **Purchase bundle:** weekday, hour, month, weekend flag and sine/cosine pairs
   for weekday/hour/month (10 columns).
2. **Purchase plus distance bundle:** the purchase bundle plus mean and maximum
   seller-to-customer distance, missing-distance fraction, log mean distance,
   and maximum distance divided by positive promised lead days (5 more columns).

Each bundle is evaluated for logistic regression, Random Forest, XGBoost and
CatBoost with unchanged hyperparameters: 48 additional fits. Previous enriched
predictions are reused after cohort/label/date alignment checks. All variants
score the same 40,056 development orders. No May–June or final holdout scoring.
This is a sequential ablation, not a full factorial experiment: distance without
purchase-time features is not tested.

## Geographic construction and limitations

The bundled Olist postcode reference contains multiple coordinates per prefix.
Records outside latitude [-35, 6] or longitude [-75, -30] are excluded as a broad
Brazil plausibility check; coordinate-wise medians provide one reference location
per postcode prefix. Haversine distance uses mean Earth radius 6,371.0088 km.

Distance is a straight-line postcode-location proxy, not actual driving distance,
carrier routing, travel time or distance from a verified fulfilment warehouse.
Multi-seller orders use one distance per unique seller, not item-weighted distances.
Mean/max exclude missing distances and the missing fraction records partial
coverage. Completely unknown distances remain missing; they are not coded as zero.
Sklearn pipelines fit numeric imputation on training rows; CatBoost handles numeric
missing values natively. Nonpositive promised lead days produce a missing ratio.

The lookup is fixed, outcome-free reference data. It uses the full geographic
archive, whose historical availability is unverified because revision dates are
absent. This is an explicit static-reference assumption, not a claim of fully
reconstructed point-in-time data. The existing related-table snapshot and
eventual-delivery population limitations still apply. No target-history features,
reviews or actual-delivery predictors are added.

## Completed results (21 September 2026)

Equal-month mean average precision, higher is better:

| Model | Previous enriched | + Purchase timing | + Purchase timing and distance |
|---|---:|---:|---:|
| Logistic regression | 0.19751 | 0.19245 | 0.19777 |
| Random Forest | 0.18505 | 0.17904 | 0.18155 |
| XGBoost | 0.18764 | 0.18676 | 0.18015 |
| CatBoost | 0.19105 | 0.18680 | 0.19115 |

Distance recovers some of the timing-only decline for logistic, forest and
CatBoost, but the complete bundle is essentially unchanged for logistic/CatBoost
relative to the previous enriched configuration and worse for forest/XGBoost.
These are observed differences, not significance claims about either bundle.
The original, unenriched logistic benchmark remains at 0.19814.

Daily top-10% coverage with both bundles is 22.23% for logistic, 20.45% for forest,
20.22% for XGBoost and 21.69% for CatBoost. These are not statistically tested
coverage gains. Logistic's AP and coverage therefore do not tell exactly the same
story; stakeholder capacity remains important.

Of 40,056 development orders, 207 have at least one seller distance missing
(about 0.52%). Median order-mean distance is 436.5 km; the 99th percentile of
order-maximum distance is 2,474.5 km. These describe the geographic proxy, not
actual shipping routes. Do not infer that plausible distances prove point-in-time
availability or causal effects.

The data do not support keeping extra features solely because they sound useful.
Distance-only ablation or controlled inner-fold tuning can be later development
experiments; adding increasingly many bundles on these same months is model
selection and requires a separate final evaluation to establish generalisation.

## Commands

After the original enriched experiment:

```bash
.venv/bin/python phase2/classification_time_distance.py
.venv/bin/python -m unittest discover -s phase2 -p 'test_*.py'
```

Outputs: `classification_time_distance_results.json` and ignored local
`classification_time_distance_predictions.csv`. Existing experiment outputs are
preserved. Eight tests pass, including coordinate aggregation, known-distance
calculations, missing handling, purchase weekday extraction and the existing
outcome/threshold/window checks. The tests do not establish snapshot availability.
