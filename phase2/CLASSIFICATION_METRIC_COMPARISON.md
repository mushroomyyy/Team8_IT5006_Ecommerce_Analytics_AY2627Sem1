# Classification metric comparison

Generated from paired development predictions; no refitting or threshold optimisation.

Ranking metrics are equal-month means. Decision/probability metrics pool orders. Capacity selection is within each day.

## Ranking and probability quality

| Model | AP ↑ | Trapezoidal PR-AUC ↑ | ROC-AUC ↑ | Brier/MSE ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|
| base_logistic_regression | 0.1981 | 0.1971 | 0.6798 | 0.1554 | 0.4856 |
| dummy_prior | 0.1053 | 0.5527 | 0.5000 | 0.0981 | 0.3629 |
| ensemble_all_four | 0.1961 | 0.1949 | 0.6803 | 0.1416 | 0.4514 |
| ensemble_logistic_catboost | 0.2012 | 0.2000 | 0.6788 | 0.1730 | 0.5223 |
| purchase_distance_catboost | 0.1912 | 0.1900 | 0.6624 | 0.1579 | 0.4878 |
| purchase_distance_logistic_regression | 0.1978 | 0.1967 | 0.6732 | 0.2261 | 0.6610 |
| purchase_distance_random_forest | 0.1816 | 0.1805 | 0.6640 | 0.1122 | 0.3822 |
| purchase_distance_xgboost | 0.1801 | 0.1789 | 0.6564 | 0.1466 | 0.4577 |

The constant-per-fold dummy illustrates why AP and trapezoidal PR-AUC must not be conflated: trapezoidal endpoint interpolation can give a misleadingly large area for constant scores. AP is the primary measure.

## Decisions under forward thresholds

| Model | Precision ↑ | Recall ↑ | F1 ↑ | Balanced accuracy ↑ | MCC ↑ | Alert fraction |
|---|---:|---:|---:|---:|---:|---:|
| base_logistic_regression | 0.1266 | 0.4133 | 0.1938 | 0.5364 | 0.0471 | 0.3483 |
| dummy_prior | 0.1033 | 0.7955 | 0.1828 | 0.4854 | -0.0235 | 0.8215 |
| ensemble_all_four | 0.1655 | 0.4915 | 0.2476 | 0.5978 | 0.1298 | 0.3168 |
| ensemble_logistic_catboost | 0.1542 | 0.5788 | 0.2435 | 0.5998 | 0.1257 | 0.4005 |
| purchase_distance_catboost | 0.1426 | 0.4196 | 0.2128 | 0.5591 | 0.0786 | 0.3140 |
| purchase_distance_logistic_regression | 0.1505 | 0.5869 | 0.2396 | 0.5957 | 0.1199 | 0.4159 |
| purchase_distance_random_forest | 0.1234 | 0.5106 | 0.1987 | 0.5386 | 0.0480 | 0.4416 |
| purchase_distance_xgboost | 0.1461 | 0.3840 | 0.2117 | 0.5580 | 0.0797 | 0.2804 |

These are model-specific forward thresholds, not matched staffing budgets. Do not optimise or select thresholds retrospectively from this table.

## Matched daily review capacity: nominal top 10%

| Model | Precision at capacity ↑ | Coverage / recall at capacity ↑ | Lift ↑ | Actual reviewed fraction |
|---|---:|---:|---:|---:|
| base_logistic_regression | 0.2244 | 0.2148 | 2.1096 | 0.1021 |
| dummy_prior | 0.1064 | 0.1018 | 1.0000 | 0.1021 |
| ensemble_all_four | 0.2284 | 0.2186 | 2.1464 | 0.1021 |
| ensemble_logistic_catboost | 0.2347 | 0.2247 | 2.2062 | 0.1021 |
| purchase_distance_catboost | 0.2267 | 0.2169 | 2.1303 | 0.1021 |
| purchase_distance_logistic_regression | 0.2323 | 0.2223 | 2.1832 | 0.1021 |
| purchase_distance_random_forest | 0.2137 | 0.2045 | 2.0085 | 0.1021 |
| purchase_distance_xgboost | 0.2112 | 0.2022 | 1.9856 | 0.1021 |

Ties use expected capture under random tie-breaking. Integer daily capacity rounds up, so actual review fraction slightly exceeds 10%.

Full confusion counts, specificity, accuracy, probability RMSE, other capacities and fold metrics are in classification_metric_summary.json.

This report adds no significance tests for the secondary metrics. Existing AP tests do not establish significance for F1, coverage or Brier. These are repeatedly inspected development cohorts, not final test results.
