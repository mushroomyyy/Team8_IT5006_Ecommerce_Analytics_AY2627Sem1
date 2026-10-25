# Model and simulation contracts

## Architecture

```text
Browser -> Streamlit Python server
           -> packaged historical inputs -> cutoff eligibility
           -> POST /predict/batch -> Render fitted C3 + R1 pipelines
           <- predictions         <- fitted preprocessing + inference
           -> risk ranking, top-X% follow-up, worklist and downloads
Explicit outcome reveal -> separate local outcome file -> retrospective metrics
```

Streamlit is the application middle layer, not merely a browser page. It handles
historical demo data, date navigation, eligibility, ranking and retrospective
metrics in its Python server. Every eligible order is scored; **Follow-up
proportion (%)** determines how many highest-risk orders are recommended for human
follow-up. The default 30% is a workload assumption, never a sample of inputs.
No fitted model or preprocessing is loaded by Streamlit.

Render provides only inference, input validation, health, metadata and aggregate
monitoring. Both services share 24 engineered approval-time features (21 numeric,
3 categorical). Query `GET /model-info` for names, categories, ranges and provenance.
The misspelling `order_frieght_value_sum` is part of the existing model contract.
No future delivery values or outcomes are model inputs.

`POST /predict` and `/predict/batch` return both model outputs. Independent
`/predict/classification`, `/predict/regression` and `/model-info/<task>` routes
keep models usable separately. `/health` succeeds only when both load.
`/logs/summary` holds bounded process-local aggregates without order IDs or inputs.
`/samples`, `/simulation`, `/worklist`, `/outcomes` and `/explain` are removed.
There is no sensitivity/explanation UI: independent feature resets could create
inconsistent calendar values and do not provide causal or intervention guidance.

The frontend batches up to 1,000 eligible rows per inference request and checks
row identity, counts and finite outputs. Only explicit API row422 errors become
`invalid_model_inputs` exclusions. Inference failures/timeouts abort the worklist
with useful errors, rather than silently excluding valid orders. Operational
results are cached by service URL, model versions, date, follow-up settings and
snapshot revision; retrospective outcomes are outside that cache and loaded only
on explicit reveal. Changing date or policy hides revealed outcomes again.

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

Metadata inside the original fitted artifacts remains provenance. Legacy threshold
fields, monthly score distributions, baselines and importance tables are excluded
from HTTP model metadata. The classifier response contains risk and input-range
warnings, without late/not-late decisions.

The historical refit was available on 2 June 2018, using known outcomes strictly
before that date and approvals from 18 April 2017 through 17 April 2018, with a
45-day waiting buffer. C3 uses 63,592 history rows; R1 uses 62,841. Source notebooks
and result tables remain under `phase2/`. The submitted report selected C3 and R1;
June guided selection, while July/August are frozen-model monitoring.

Historical June C3 AP is about 11.22%; R1 RMSE is about 7.369 days and MAE 5.81 days.
The report's monthly top-30% captured 76/137 known late orders (~55.5%). Those are
Phase 2 cohort results, not a promise for the implemented daily filtered policy.

## Daily simulation

- Current dates: **2 June–30 August 2018**; batch date: previous calendar day.
  June guided model selection; July and August are frozen-model monitoring months.
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
orders. `frontend/data/simulation_orders.json` has engineered features, approval, promise
and delivery timestamps for as-of logic, never final status. These stay in the
Streamlit Python server; only order identifiers and engineered features are sent
to Render. Displayed worklist rows omit delivery timestamps.
`frontend/data/simulation_outcomes.json` is separate and never read for eligibility
or scoring. Its labels follow Phase 2's
approval-day-plus-45-days inclusive evaluation rule, preserving unknowns.

## Retrospective daily evaluation

Outcomes remain hidden until an explicit action in Streamlit reads the separate outcome file and evaluates
the **same already-scored worklist**, without another prediction request. Metrics report totals, known, unknown and late counts for
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

The documented inference request accepts exactly the feature names listed by
`/model-info`. A complete synthetic example is in
[`../scripts/verify_http.py`](../scripts/verify_http.py); it submits no real order
identifiers or copied transaction records. To verify a local or deployed API:

```bash
python phase3/scripts/verify_http.py https://team8-delivery-api.onrender.com
```

The verifier checks integrated vs independent model outputs, single/batch numerical
equivalence, invalid single/empty-batch422 errors, Swagger and404 responses for
removed demo/explanation paths. Last-bit linear algebra differences are compared
with `rtol=atol=1e-12`.

A minimal client, given a valid 24-feature dictionary:

```python
import requests
base = 'https://team8-delivery-api.onrender.com'
response = requests.post(base + '/predict', json={'features': features}, timeout=60)
response.raise_for_status()
result = response.json()
print(result['classification']['late_probability'])
print(result['regression']['predicted_days_from_promise'])
```

For multiple orders, submit `{'rows': [{'order_id': 'synthetic-1',
'features': features}]}` to `/predict/batch`. Results retain input index and ID;
invalid rows carry errors without changing the alignment of valid predictions.
Historical dates and outcome reveal are Streamlit operations, not API endpoints.

## Frontend request handling

- Each simulated day is scored once through `/predict/batch` (chunks of 100,
  5 s / 60 s timeouts) and cached per API URL, model versions, date and data
  revision. The follow-up proportion and minimum days re-rank locally without
  API calls. Scored rows are every order undelivered at the cutoff with a
  promise date, a superset of the eligible rows for any minimum-days setting.
- The score cache is a process-wide dictionary rather than `st.cache_data`, so a
  refreshed page or a second viewer never waits on another session's request.
- The API is checked once per session (and on Retry) by polling `/health` with
  short calls, showing elapsed time instead of one long blocking request.

## Month-end review

- `phase3/scripts/score_replay.py` pre-scores all 18,960 June–August replay
  orders with the packaged C3/R1 artifacts (identical to the deployed API per
  `render_model_equivalence.json`) into `frontend/data/replay_scores.json`.
- Each month replays every morning with the daily list rules; outcomes are
  attached only after ranking. Comparisons: whole-month C3 ranking (Phase 2
  style), random calls, and the same daily rules ranked by R1's predicted days
  from promise.
- `frontend/data/simulation_timing.json` holds R1's Phase 2 target per order
  (`attach_regression_actuals`, 45-day cap, final canceled/unavailable excluded).
  June reproduces the reported R1 RMSE 7.369 and MAE 5.81.

