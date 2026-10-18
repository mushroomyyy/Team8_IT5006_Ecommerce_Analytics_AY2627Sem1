# C3 classification deployment

This standalone deployment packages the Phase 2 tuned Random Forest classifier.
It does not import, start, or depend on the regression deployment. Existing
Phase 1 Streamlit entry points are unchanged. Run commands from the repository root.

## What is included

- `deployment/train_classification.py`: offline export using the Phase 2 feature
  builder, reference-coded categorical preprocessing, median numeric imputation,
  and cached CV-tuned C3 parameters. No linear-model log/cyclic transforms.
- `deployment/artifacts/classification_c3_rf.joblib`: fitted preprocessing and
  classifier together; serving does not need raw CSVs or a training run.
- Classification metadata: feature schema, categories, dependency versions,
  training fingerprint, parameters and historical evaluation.
- Thirty held-out June feature-only examples, with zero overlap with refit history.
- `deployment/classification_api.py`: independently hosted FastAPI service.
- `deployment/classification_streamlit.py`: HTTP frontend for that API.

The exported model fits 63,592 known-outcome history orders before the Phase 2
run date, 2 June 2018. The exporter scores the same June eligibility cohort as the
notebook and asserts AP agreement within 1e-10 (reference 0.11215692321859022).
It also checks identical probabilities before/after serialization. June outcomes
are used for this reproduction check, never fitting or threshold optimization.

## Backend locally

Use Python **3.13.5**, the serialization environment recorded by the artifact.
Use a separate virtual environment from other course deployment examples.

```bash
python3.13 -m venv .venv-classification-api
source .venv-classification-api/bin/activate
python -m pip install -r deployment/classification_requirements.txt -c deployment/classification_backend_constraints.txt
python -m uvicorn deployment.classification_api:app --host 127.0.0.1 --port 8002
```

Open http://127.0.0.1:8002/docs for Swagger UI. `/health` returns HTTP 200 only
after the pipeline loads, or HTTP 503 when unavailable. `/model-info` exposes
the artifact metadata. `POST /predict` accepts `{"features": {...}}`; the Swagger
example is complete and executable. Classification-suffixed aliases are also
available at `/predict/classification` and `/model-info/classification`.

Example using a checked-in held-out order (another terminal, same environment):

```bash
python - <<'PY'
import json
from urllib.request import Request, urlopen
with open('deployment/artifacts/classification_sample_orders.json') as f:
    row = json.load(f)[0]
row.pop('order_id')
request = Request('http://127.0.0.1:8002/predict',
                  data=json.dumps({'features': row}).encode(),
                  headers={'Content-Type': 'application/json'})
with urlopen(request, timeout=60) as response:
    print(json.load(response))
PY
```

The response includes `late_probability`, `is_predicted_late`, `threshold`,
`threshold_policy`, `model_code`, and `model_version`.

All 24 engineered feature keys are required. Numeric nulls use the fitted
training median; categorical nulls and unknown strings are rejected with HTTP
422 and the trained allowed values. Matching is exact: `RJ` is valid, `rj` is
not. New payment methods or categories require a reviewed retraining procedure;
the current model has no learned Other category. A learned unknown-category
policy or frequency-weighted sensitivity is future work, not a silent default.
Numeric strings, booleans, nonfinite values and magnitudes above 1e12 are
rejected. Item, seller, product-category, seller-state counts and payment
installments must be integers >=1; preceding-week approvals must be >=0.
Promised lead days must be integer, may be negative, and can fall outside the
refit range (-6 to 154) within the sanity limit. Money, dimensions, distances,
photos, ratios and approval lag must be nonnegative. Weekday encoding is
**Monday=1 through Sunday=7**; weekend flags must match weekdays 6/7 and
December flags must match month 12 when both fields are supplied. Invalid
day/month combinations are rejected (February29 allowed because no year is
supplied). Flags are numeric 0/1. Values outside refit 1st–99th percentiles
produce input-drift warnings, not a claim of measured population drift. These are
engineered model inputs: manual clients must reproduce the Phase 2 aggregation
and as-of feature definitions rather than submit an unprocessed order record.

## Frontend locally

In another terminal, use a separate Python 3.13 environment:

```bash
python3.13 -m venv .venv-classification-ui
source .venv-classification-ui/bin/activate
python -m pip install -r deployment/classification_ui_requirements.txt -c deployment/classification_ui_constraints.txt
CLASSIFICATION_API_URL="http://127.0.0.1:8002" python -m streamlit run deployment/classification_streamlit.py
```

The UI sends HTTP requests to FastAPI; it does not load the model locally. The
order selector resets feature values when switching records. The performance
tab displays historical AP, ROC-AUC, top-10% lift and monitoring-period metrics.
Connection and validation errors are displayed in the interface.

**Threshold interpretation:** 0.5 is an illustrative single-order cutoff, not a
validated operational alerting policy. It had zero June recall. A below-cutoff
result is not an assurance of on-time delivery. Top-10% evaluation ranks a cohort;
it is not interchangeable with this cutoff. The model is a deployment candidate
with the highest June AP, not a winner across every metric or later period.

## Render: two independent services

Create services from this branch with repository root as the service root.
Set `PYTHON_VERSION=3.13.5` on both services. No credentials are needed.

| Setting | Classification backend | Optional classification frontend |
|---|---|---|
| Build | `pip install -r deployment/classification_requirements.txt -c deployment/classification_backend_constraints.txt` | `pip install -r deployment/classification_ui_requirements.txt -c deployment/classification_ui_constraints.txt` |
| Start | `python -m uvicorn deployment.classification_api:app --host 0.0.0.0 --port $PORT` | `python -m streamlit run deployment/classification_streamlit.py --server.address 0.0.0.0 --server.port $PORT` |
| Health path | `/health` | `/_stcore/health` |
| Environment | `PYTHON_VERSION=3.13.5` | `PYTHON_VERSION=3.13.5`, `CLASSIFICATION_API_URL=https://YOUR-CLASSIFICATION-API.onrender.com` |

Use the committed pipeline for ordinary deployment; do not train on every web
service startup/build. Check `/health`, `/docs`, `/model-info` and one prediction
on the deployed URL before sharing it. A deployment may need time to start;
the frontend allows a 60-second request timeout. Hosting has not been provisioned
by this change. Regression can deploy independently using its own entry point.

## Rebuild and verify

Offline export needs the bundled Olist archive and the `phase2` code/results.
The packaged API and frontend do not extract or load the complete dataset.

```bash
python -m deployment.train_classification
python -m pip install httpx==0.28.1
python -m unittest deployment.test_classification deployment.test_classification_api -v
```

Frontend tests use JSON fixtures and mocked HTTP only; run them in the
frontend-only environment, which must not have sklearn or joblib installed:

```bash
FRONTEND_ONLY_TEST=1 python -m unittest deployment.test_classification_ui -v
```

Do not replace or tune the artifact using June outcomes. Future pipeline changes
require retraining, metadata/version updates and the same reproduction checks.


## Operations demo and additional endpoints

The UI serves the operations/logistics team monitoring newly approved orders.
Use its navigation to check an order, replay a June day, understand scores,
inspect historical performance/monitoring, or read the model card. The structured
24-feature form provides trained-category dropdowns, numeric missing-value
controls, field definitions, and range warnings. Raw JSON is an advanced option.
Samples come from the API, so the frontend does not supply independently
versioned prediction inputs. Metadata is cached for five minutes; retry clears
it. The sidebar reads backend readiness. The root `.streamlit/config.toml`
provides the same orange/navy theme as Phase 1.

| Endpoint | Purpose |
|---|---|
| `GET /samples` |30 samples spread over all June dates, chosen without outcomes |
| `GET /replay?day=2018-06-15` |All eligible feature-only orders for one June day; omit day to list dates |
| `GET /outcomes?day=2018-06-15` |Separate historical display-only outcomes; unknown outcomes are null |
| `POST /predict/batch` |Up to 1000 rows, each `{order_id?, features}`; row-level validation errors |
| `POST /explain` |Same request as `/predict`; one-feature-at-a-time sensitivity against history medians/modes |
| `GET /logs/summary` |Bounded process-local monitoring; resets on restart, no order IDs or input values stored |

Batch requests use `{"rows":[{"order_id":"optional-id","features":{...}}]}`.
The overall response is 200 with a `results` entry for each row, containing either
`prediction` or a row-level `error`/`status_code`. Invalid request shape or more
than 1000 rows returns 422. Unexpected model failures return generic 500 messages
and detailed server logs; client validation failures return 422.

The fixed June top 10% score boundary is derived from scores alone, not outcomes.
It is a historical reference; daily watch-list selection ranks the replayed day's
orders independently. The app distinguishes that ranking from both the June
reference and the illustrative 0.5 decision cutoff. Percentiles and base-rate
comparisons are historical context, not current calibrated guarantees.

Explanations show how the probability changes when each individual feature is
reset to its history median (or categorical mode). These are sensitivity probes,
not causal effects. Independent resets can break relationships between calendar
features; they are not complete alternative orders. What-if grids must still
pass the normal API contract. Uploads require the exact engineered schema.
Historical outcomes are fetched only for display and capture calculation, with
unknown outcomes excluded from outcome denominators. They never fit/tune the
model or choose demonstration rows or reference score thresholds.

Optional environment settings for UI links: `CLASSIFICATION_REPORT_URL` for the
final report and `REGRESSION_APP_URL` for the separately deployed regression app.
Neither is needed for classification to run. Do not place credentials in URLs.

## Reproducible Linux container checks

Constraints were resolved on Linux/amd64 with Python 3.13.5. Keep the requirements
and corresponding constraints together in all installs. Dockerfiles use the
same Render start commands and include the root Streamlit theme:

```bash
docker build --platform linux/amd64 -f deployment/Dockerfile.classification-api -t c3-classification-api .
docker build --platform linux/amd64 -f deployment/Dockerfile.classification-ui -t c3-classification-ui .
docker network create c3-demo
docker run --rm --name c3-api --network c3-demo -p 8002:10000 c3-classification-api
docker run --rm --network c3-demo -p 8501:10000 -e CLASSIFICATION_API_URL=http://c3-api:10000 c3-classification-ui
```

Start the API and UI in separate terminals; use the backend container name on
the Docker network, not localhost, for the frontend API URL. Native Render
services instead use the deployed HTTPS URL. Verify memory, cold starts,
HTTPS/proxy behaviour and service health timing on the actual chosen Render
tier; a successful local container test does not provision hosting.

The exporter checks the notebook fitted-state fingerprint after normalizing
`n_jobs` to the notebook value, and stores actual/normalized/reference hashes.
The fitted model version remains unchanged for metadata-only additions. June
replay features and outcomes are shipped in separate files. Metrics preserve
the CSV floating-point precision. No regression source files or shared Phase 1
entry points are modified by this branch. After both branches merge, the owner
of the shared deployment overview should update its regression-only wording
and link this document.

Monitoring request counts cover `/predict` and `/predict/batch` HTTP calls;
successful prediction counts cover scored rows. Explanation probes are excluded.
Counts are per process, not durable or shared across workers.

## Review verification (19 October 2026)

- 36 host tests: 8 model/export tests, 15 API tests and 13 frontend tests.
- Backend and frontend build from Python 3.13.5-slim Linux/amd64 with constraints.
- Frontend-only container tests assert sklearn/joblib cannot be imported.
- Real HTTP container demo: check order, explain 24 sensitivities, replay day,
  open ranked order, compare scenarios, reveal backtest, view performance.
- Notebook-normalized fitted fingerprint and June AP remain unchanged.
- Local emulated-container memory after the demo: about 197 MiB backend and 65 MiB
  frontend; actual hosting hardware, cold starts and concurrent loads vary.
- Mobile-width page screenshots are in [classification_screenshots](classification_screenshots/).
- Real Render hosting has not been provisioned or verified.
