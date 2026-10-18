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
python -m pip install -r deployment/classification_requirements.txt
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

All 24 engineered feature keys are required; explicit nulls use the pipeline's
missing-value handling. Numeric strings, infinities, unknown feature names,
invalid calendar/count values and negative physical quantities are rejected
with HTTP 422. Weekday encoding is **Monday=1 through Sunday=7**. Unknown
categorical strings follow the fitted encoder's ignore policy. These are
engineered model inputs: manual clients must reproduce the Phase 2 aggregation
and as-of feature definitions rather than submit an unprocessed order record.

## Frontend locally

In another terminal, use a separate Python 3.13 environment:

```bash
python3.13 -m venv .venv-classification-ui
source .venv-classification-ui/bin/activate
python -m pip install -r deployment/classification_ui_requirements.txt
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
| Build | `pip install -r deployment/classification_requirements.txt` | `pip install -r deployment/classification_ui_requirements.txt` |
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
python -m unittest deployment.test_classification deployment.test_api -v
```

For the optional frontend tests, also install the UI requirements and run:

```bash
python -m unittest deployment.test_classification_ui -v
```

Do not replace or tune the artifact using June outcomes. Future pipeline changes
require retraining, metadata/version updates and the same reproduction checks.
