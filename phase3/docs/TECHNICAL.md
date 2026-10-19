# Model and simulation contracts

## Architecture

```text
Streamlit -> HTTP FastAPI -> C3 fitted preprocessing + Random Forest
                         -> R1 fitted preprocessing + OLS
                         -> daily as-of eligibility and deterministic risk ranking
Retrospective outcomes -> explicit outcome endpoint only
```

The frontend never deserializes models or reads artifacts. Both services share
24 engineered approval-time features (21 numeric, 3 categorical). Query
`GET /model-info` for names, trained categories, ranges and model provenance.
The misspelling `order_frieght_value_sum` is part of the existing model contract.
No future delivery values or outcomes are model inputs.

`POST /predict` and `/predict/batch` return both model outputs. Independent
`/predict/classification`, `/predict/regression` and `/model-info/<task>` routes
keep the two models usable separately. `/health` succeeds only when both load.
`/explain` provides one-feature sensitivity resets, not causal explanations.
`/logs/summary` holds bounded process-local aggregates without order IDs or inputs;
simulation/explanation probes do not enter these direct prediction counters.

## Selected fitted models

- **C3**: tuned Random Forest, 500 trees, entropy, depth 8, leaf minimum 2,
  split minimum 50, `max_features=log2`, bootstrap fraction 0.7, no class weighting.
  Numeric median imputation and reference-coded one-hot encoding are fitted
  inside its pipeline. Risk is the positive-class score, without a fixed 0.5
  decision or a calibration guarantee. No monthly percentile decides follow-up.
- **R1**: all-feature OLS with fitted log1p transformations, cyclic month/weekday,
  mixed-route merging, numeric imputation/scaling and reference encoding.
  Target: `min(delivery_date - approval_date, 45) - (promise_date - approval_date)`
  in calendar days. Negative values mean early relative to promise, positive late.
  This is a capped delivery offset, not an uncapped duration or guaranteed arrival.
  The serving wrapper clips raw predictions to `[-promised_lead_days,
  45-promised_lead_days]`, matching Phase 2 evaluation. Raw output is also exposed.
  If promised lead is null, fitted imputation supports the raw estimate but the
  wrapper cannot apply this bound; the result explicitly explains that limitation.

Both artifacts preserve fitted preprocessing. C3 bytes are moved unchanged from
the pre-integration working tree. R1 is not retrained: only its custom transformer
class import path changes to `phase3.backend.transforms.ExpandColumns`, so inference
works without Phase 2 imports. `regression_packaging.json` records source and
packaged SHA256 hashes and exact pre/post serialization equality for all 6,164
June rows. `manifest.json` hashes the current packaged files.

Metadata inside the original fitted artifacts remains provenance: old threshold
fields do not define the integrated application's policy. The deployed classifier
response contains risk and input-range warnings, without late/not-late decisions.

The historical refit was available on 2 June 2018, using known outcomes strictly
before that date and approvals from 18 April 2017 through 17 April 2018, with a
45-day waiting buffer. C3 uses 63,592 history rows; R1 uses 62,841. Source notebooks
and result tables remain under `phase2/`. The submitted report selected C3 and R1;
June guided selection, while July/August are frozen-model monitoring.

Historical June C3 AP is about 11.22%; R1 RMSE is about 7.369 days and MAE 5.81 days.
The report's monthly top-30% captured 76/137 known late orders (~55.5%). Those are
Phase 2 cohort results, not a promise for the implemented daily filtered policy.

## Daily simulation

- Current dates: **2 June–1 July 2018**; batch date: previous calendar day.
  The cohort is approved orders, not purchase dates or a cumulative unresolved queue.
- Source timestamps have no offset. Treat them as Brazilian local wall time
  (America/Sao_Paulo convention), without conversion to Singapore/UTC.
- Cutoff: midnight at the start of current date. Only delivery strictly before
  cutoff supports `already_delivered` exclusion. Later delivery does not affect
  eligibility, scores, ranking or the worklist response.
- Missing promised dates are excluded explicitly. Orders whose promise date minus
  current date is **less than or equal to** the configured minimum days remaining
  are excluded. Default 1 means exclude due today, tomorrow or earlier.
- Cancellation timestamps are absent. Final canceled/unavailable status never
  drives as-of eligibility. R1 training/evaluation excluded these final-status
  rows; this population difference is a disclosed limitation.
- Validate model inputs and expose per-order exclusions for invalid inputs.
- Rank eligible orders by C3 risk descending, then order ID ascending for ties.
  Select `ceil(review_proportion * eligible_count)`; zero selects none and positive
  proportions can select one order in a small batch. Default 30% is a demo setting.
  Exclusion reason counts may overlap; excluded counts always count distinct orders.
- Going backwards recomputes cutoff knowledge; no future statuses persist.
  Promise/catalogue/item/payment/seller features are final snapshots whose changes
  cannot be reconstructed. This is a historical simulation with those limitations.

Snapshots are exported offline from the same Phase 2 feature builder and source
orders. `simulation_orders.json` has engineered features, approval, promise and
delivery timestamps for as-of logic, never final status. Delivery timestamps are
private to the server; worklist outputs omit them. `simulation_outcomes.json` is
separate and never read for eligibility or scoring. Its labels follow Phase 2's
approval-day-plus-45-days inclusive evaluation rule, preserving unknowns.

## Retrospective daily evaluation

Outcomes remain hidden until an explicit action calls `/outcomes` with the same
policy parameters. Metrics report totals, known, unknown and late counts for
**all**, **eligible** and **selected** populations. Selected late rate uses known
selected outcomes; eligible coverage uses known late eligible orders; all-order
coverage uses all known late orders, including excluded orders. Zero denominators
return null. Unknowns are not silently treated as on-time. The application makes
no claim that June is untouched validation of the chosen daily assumptions.

## Validation and reproduction

Every numeric input must be finite and within magnitude 1e12; booleans, impossible
calendar combinations, inconsistent derived flags and unsupported categories
are rejected with useful errors. Numeric nulls use fitted imputation; all-null
numeric rows are rejected. Batch requests accept 1–1,000 rows and report row errors.

Export utilities require the original source tables/bundled ZIP and Phase 2 modules:

```bash
python -m phase3.export.train_classification  # offline refit; not required to serve
python -m phase3.export.train_regression      # offline refit; not required to serve
python -m phase3.export.package_regression    # repackage original remote-branch artifact, no fit
python -m phase3.export.export_simulation     # snapshot export, no model fit; refresh manifest last
```

Do not retrain casually: the checked-in fitted models and export provenance are
the intended deployment inputs. Original regression branch is preserved in Git.
The separate course technical HTML guide could not be located; course PDF and
professor deployment examples were reviewed instead.

## API examples

A single prediction and a mixed valid/invalid batch, using packaged sample inputs:

```python
import json
from urllib.request import Request, urlopen

base = 'http://127.0.0.1:8002'  # replace with the verified Render HTTPS URL
sample = json.load(urlopen(base + '/samples'))['samples'][0]
features = {k: v for k, v in sample.items() if k != 'order_id'}
def post(path, payload):
    request = Request(base + path, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    return json.load(urlopen(request, timeout=90))

single = post('/predict', {'features': features})
print(single['classification']['late_probability'])
print(single['regression']['predicted_days_from_promise'])
batch = post('/predict/batch', {'rows': [
    {'order_id': sample['order_id'], 'features': features},
    {'order_id': 'invalid-example', 'features': {}},
]})
print(batch['successful_rows'])  # 1; invalid row carries a 422 error
```

A daily request with explicit assumptions:

```bash
curl 'http://127.0.0.1:8002/worklist?simulation_date=2018-06-02&review_proportion=0.3&minimum_days_remaining=1'
curl 'http://127.0.0.1:8002/outcomes?simulation_date=2018-06-02&review_proportion=0.3&minimum_days_remaining=1'
```

The second call explicitly reveals retrospective outcomes. Its population counts
and rates are separate from the operational worklist. Run
`python phase3/scripts/verify_http.py https://ACTUAL-API-URL` after deployment to
check health, Swagger, both date boundaries, invalid requests and single/batch equality.
