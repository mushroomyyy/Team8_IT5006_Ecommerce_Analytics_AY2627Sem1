# Feature engineering, CatBoost and statistical comparison

Completed 21 September 2026. This is development evidence, not independent final testing.

## What was run

Eight variants cover original/enriched features for logistic regression, Random
Forest, XGBoost and CatBoost. All use the same 40,056 validation orders across
six monthly folds, November 2017–April 2018. Training uses the existing moving
180-day window, 30-day gap and label-maturity checks. May–June was not rescored.
Original logistic/forest/XGBoost predictions were reused from the corrected
backtest after cohort/label checks. Thirty additional fits cover original
CatBoost and the four enriched models. Hyperparameters were fixed, not tuned.

CatBoost 1.2.10 used 300 iterations, depth 6, learning rate 0.05, balanced class
weights and seed 42. Categorical columns were supplied natively, with an explicit
missing token; numeric missing values were left for CatBoost to handle. No scoring
fold was used for early stopping or categorical-statistic fitting. Other models
retain fold-fitted imputation/encoding and their original configurations.
See [CatBoost's categorical-feature documentation](https://catboost.ai/docs/en/features/categorical-features)
and [the original paper](https://papers.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf).

## Added features and leakage boundaries

Nineteen additional predictors (16 numeric, 3 categorical):

- Purchase-to-approval hours and promised-delivery weekday.
- Price, freight, weight and volume per item; weight/volume and price/weight.
- Multiple-seller indicator and total-weight-missing indicator.
- Sine/cosine encodings of approval hour, weekday and month.
- Category of the highest-price item, largest-payment type and customer/seller state route.

Ties in primary item/payment selection use recorded sequence numbers. All these
inputs are plausibly available at approval; actual deliveries, reviews, shipping
handoff timestamps, order status and target-derived history are excluded.
The outcome-removal test confirms identical features after removing delivery/status
columns. This does **not** prove historical availability of related-table values:
Olist has no revision history. The eventual-delivery population restriction also
remains; this is not yet a production-valid evaluation of all approved orders.

## Results

AP is average precision, not trapezoidal PR-AUC. Each validation month receives
equal weight in the mean. The coverage column pools captured late orders across
daily top-10% selections and is a different estimand.

| Model | Original mean AP | Enriched mean AP | Enriched daily coverage at 10% |
|---|---:|---:|---:|
| Logistic regression | 0.19814 | 0.19751 | 22.05% |
| Random Forest | 0.17081 | 0.18505 | 20.85% |
| XGBoost | 0.18447 | 0.18764 | 21.02% |
| CatBoost | 0.18300 | 0.19105 | 21.48% |

Original logistic regression captures 21.48% at the same daily capacity.
No uncertainty test for coverage was performed in this experiment. Better AP
does not automatically imply better operational coverage or calibrated probabilities.
For example, enriched logistic Brier score worsens from 0.1554 to 0.2248 despite
similar AP. Ranking gains must not be interpreted as probability-quality gains.

## Statistical procedure

Paired circular moving-block bootstrap resamples dates within each validation
month, using identical date weights for every model. All orders on a sampled date
move together; missing calendar days remain in the resampling grid. Weighted AP
is recomputed within each month and averaged equally across the six months.
The primary block length is seven days, with three- and fourteen-day sensitivity
analyses, each using 1,999 replicates and seed 42. The primary length and comparison
family were chosen before inspecting these p-values, but after earlier pilot results.

Approximate two-sided p-values compare the observed AP difference with the
null-centered bootstrap differences. Holm correction covers ten comparisons:
seven candidates against original logistic plus enriched-versus-original
forest, XGBoost and CatBoost. Logistic's own feature ablation is already in the
first seven. Correction is applied separately within each block-length analysis;
do not select whichever sensitivity result is most favourable. Intervals below
are marginal percentile intervals, not simultaneous intervals.

### Do the extra features help each model?

Positive differences favour enrichment. Seven-day results:

| Model | AP difference | Marginal 95% interval | Holm-adjusted approximate p |
|---|---:|---:|---:|
| Logistic regression | −0.00063 | [−0.00519, 0.00397] | 0.7985 |
| Random Forest | +0.01424 | [0.00775, 0.02056] | 0.0050 |
| XGBoost | +0.00317 | [−0.00289, 0.00907] | 0.5880 |
| CatBoost | +0.00805 | [0.00356, 0.01270] | 0.0075 |

The positive forest and CatBoost differences remain significant at 0.05 across
all three block lengths. XGBoost and logistic feature gains are not established.
Failure to reject is not evidence of equivalence.

### Do the tree models beat original logistic regression?

No. Negative differences favour logistic. Seven-day results:

| Candidate | AP difference vs logistic | Marginal 95% interval | Holm-adjusted approximate p |
|---|---:|---:|---:|
| Original Random Forest | −0.02733 | [−0.03418, −0.01997] | 0.0050 |
| Original XGBoost | −0.01367 | [−0.02161, −0.00597] | 0.0070 |
| Original CatBoost | −0.01514 | [−0.02069, −0.00947] | 0.0050 |
| Enriched Random Forest | −0.01310 | [−0.02114, −0.00533] | 0.0070 |
| Enriched XGBoost | −0.01050 | [−0.01690, −0.00390] | 0.0120 |
| Enriched CatBoost | −0.00710 | [−0.01210, −0.00197] | 0.0195 |

Logistic's advantage over enriched CatBoost is **sensitive to block length**:
adjusted p=0.069 with three-day blocks, 0.0195 with seven days and 0.012 with
fourteen days. Do not call that comparison decisively settled. Other listed
logistic-versus-tree comparisons remain below 0.05 across these sensitivity runs.
These tests do not establish whether XGBoost's previously observed May–June AP
advantage was significant; that period was not tested here.

## Interpretation and limits

The feature bundle helps Random Forest and CatBoost, but logistic remains the
mean-AP leader in this development experiment. CatBoost is the closest enriched
tree challenger by observed mean AP, not a statistically established best tree.

Inference is conditional on the fitted models and these six months. It assumes
within-month dependence is reasonably captured by the chosen blocks; it does not
resample training runs or entire new months, capture arbitrary cross-month
dependence, or correct adaptive feature/model search. Monthly strata are fixed,
so these intervals are not uncertainty about performance in a new month. No
universal model-superiority claim is justified. Hyperparameter tuning, calibration,
and independent final evaluation remain outstanding. DM has not been run; a
supplementary daily additive-loss test would answer a different question than AP.

## Report-ready paragraph

We compared four classifier families/configurations using the original predictors
and a 19-feature enrichment bundle under six chronological moving-window folds.
The enrichment increased mean average precision for Random Forest from 0.1708 to
0.1850 and CatBoost from 0.1830 to 0.1910. A paired, month-stratified seven-day
block bootstrap produced Holm-adjusted approximate p-values of 0.0050 and 0.0075
for these respective gains; both conclusions were retained with three- and
fourteen-day blocks. Original logistic regression retained the highest mean AP
(0.1981). Its advantage over enriched CatBoost was sensitive to the block-length
assumption. These findings are exploratory and conditional on the fitted models
and development months; they do not replace an independent final evaluation.

For the assignment's family count, describe logistic as linear and RF/XGBoost/
CatBoost as tree-based ensemble approaches, rather than adding four unrelated
problem/model sections. Check the course's intended definition of model family.

## Next implementation stage

1. Retain original logistic as the benchmark and enriched CatBoost as a challenger.
2. Use inner chronological splits to tune regularisation/weights for logistic and
   depth, leaf regularisation and iterations for CatBoost; keep outer folds for scoring.
3. Ablate feature groups to identify which additions help; do not claim the whole
   bundle's gain comes from any single feature yet. Keep these as development analyses.
4. Assess calibration and unweighted alternatives using earlier mature labels only.
5. Freeze population, features, settings and operational capacity before independent
   evaluation; do not relabel already inspected months as an untouched holdout.

Seven automated tests pass, including outcome-independent enrichment, CatBoost
input handling, weighted AP with ties against scikit-learn, identical-prediction
null behaviour and Holm correction. Reproduction commands are in `README.md`.
