# Evaluation metrics: what each answers

## Classification: will this order be late?

The existing primary metric is **average precision (AP)**, not precision at one
threshold. AP summarises the precision–recall trade-off across score thresholds.
It is a recall-increment-weighted average of precision values. It is not identical
to trapezoidal integration of a PR curve; interpolation can make the latter too
optimistic. Report the precise implementation, not an ambiguous “PR-AUC/AP”.
See [scikit-learn's AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

| Purpose | Metrics | What they answer |
|---|---|---|
| Primary ranking | Equal-month mean AP; monthly AP and prevalence | Do late orders tend to receive higher scores, emphasising precision/recall? |
| Additional ranking | ROC-AUC; explicitly labelled trapezoidal PR-AUC | How well are classes separated across thresholds? PR-AUC here is supplementary, not a replacement selected after results. |
| Decisions at a fixed policy | Precision, recall, F1, confusion matrix | How many alerts are useful, how many late orders are caught, and how many are missed? |
| Balanced decision summaries | Balanced accuracy, MCC; specificity | How does the thresholded classifier perform beyond majority-class accuracy? |
| Probability quality | Brier score and log loss; reliability plots | Are predicted risks useful as probabilities, rather than merely rankings? |
| Daily operational capacity | Precision@capacity, recall@capacity (coverage), lift | At the same staffing budget, how many reviewed orders are actually late and how many late orders are captured? |

**Precision** = TP/(TP+FP); **recall** = TP/(TP+FN). A system can achieve high
precision by issuing very few alerts while missing nearly all late orders.
F1 combines precision and recall but does not represent a stakeholder-specific
cost function. Accuracy alone can reward predicting the majority class everywhere.
Balanced accuracy averages sensitivity and specificity; MCC uses all four
confusion cells. Report false positives/negatives as counts alongside rates.

AP is not invariant to prevalence: compare models on the same orders and report
each month's late rate. For a constant-score classifier AP equals prevalence.
Pooled scores from monthly refits can also rank months by their differing risk;
equal-month AP and daily capacity metrics help distinguish this from within-day
prioritisation. ROC-AUC remains useful but does not directly express alert precision
at the minority-class prevalence. Do not call it universally invalid for imbalance.

For our stakeholder, retain mean AP as the development ranking criterion and
daily precision/coverage at the agreed capacity as the operational assessment.
Do not search all metrics and claim whichever one happens to favour a candidate.
The 10% daily budget is still provisional, not a confirmed stakeholder requirement.

## Where MSE, RMSE and MAE fit

### Binary probabilities

For labels y in {0,1} and late probabilities p:

`Brier score = mean((p - y)^2) = binary probability MSE`.

Probability RMSE is just the square root of Brier and gives exactly the same
model ordering on the same sample. It adds no independent evidence. We include
it in the JSON for clarity but do not fill the report with duplicate metrics.
Probability MAE is not a proper probability-scoring rule: its expected value
favours extreme majority-class decisions rather than the true event probability.
Use Brier/log loss for probability quality, supplemented by reliability plots.
Neither Brier nor log loss measures calibration alone; discrimination and outcome
uncertainty also affect them. See [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

For hard binary predictions, MAE and MSE both equal the misclassification rate
(1−accuracy); RMSE is its square root. They do not solve the imbalance problem.

### Regression / forecasting, if adopted as the second problem

| Target | MAE | RMSE | Other useful reporting |
|---|---|---|---|
| Delivery duration/date | Typical absolute error in days | Error in days, emphasising large mistakes | R²; median absolute error if useful |
| Daily late-parcel count | Absolute count error per day | Count error with stronger penalty for large misses | Naive/seasonal-naive comparison; forecast horizon; R² with caution |

MSE is in squared units (days² or parcels²) and contains the same ranking
information as RMSE on the same sample. Do not calculate regression RMSE on binary
labels and interpret it as “days wrong” or “daily forecast error”. Summing order
probabilities is not automatically a valid count forecast: order availability,
forecast origin/horizon and probability calibration must first be defined.
Avoid MAPE if daily counts can be zero. Regression metrics are not implemented
for a new second problem because that target has not yet been agreed.

## Reporting and statistical comparison

`classification_metric_report.py` evaluates saved forward predictions without
refitting models or choosing new thresholds. It adds explicit trapezoidal PR-AUC,
balanced accuracy, MCC, specificity, confusion counts, alert fraction, probability
RMSE and precision at daily capacity. The prior AP/ROC-AUC/precision/recall/F1/
Brier/log-loss/coverage metrics were already present; they were not missing from
the original evaluation, only underemphasised in the concise summaries.

The output separates equal-month ranking summaries, pooled prospective-threshold
decisions and matched daily capacity. Mean monthly precision/F1 is not the same
as precision/F1 from pooled confusion counts; label aggregation explicitly.
Do not reselect thresholds on these results. Accuracy is supplementary only.

Existing bootstrap p-values test AP differences only. They cannot be reused to
claim significant F1, recall, Brier or coverage differences. Choose a small set
of decision-relevant comparisons before further testing, preserve pairing/time
dependence and account for multiplicity. Report effect sizes and uncertainty,
not just p-values. This remains repeatedly inspected development evaluation.

## Report-ready methodology paragraph

Given the imbalanced late-delivery outcome, average precision was retained as the
primary ranking metric and reported with monthly prevalence and cross-validation
variation. ROC-AUC and explicitly labelled trapezoidal PR-AUC were supplementary.
Precision, recall, F1, balanced accuracy, MCC and confusion counts characterised
decisions under prospectively selected thresholds. Brier score and log loss
assessed probability quality, while daily precision, coverage and lift at fixed
review capacities assessed operational usefulness. Metrics were computed from
paired out-of-fold predictions without retrospective threshold optimisation.
Statistical claims were limited to the separately specified AP comparisons;
additional metrics were descriptive. Independent final evaluation remains pending.
