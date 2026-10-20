# Phase 3: delivery operations

Every morning, operations receives a prioritised list of yesterday's approved
orders, supported by C3 classification risk and R1 regression estimates.
Streamlit’s server-side Python code loads the historical demo, applies eligibility
and ranks results returned by one FastAPI model-inference service over HTTP.

## Run locally

Use Python 3.13.5, matching the tested deployment images and fitted artifacts.
From the repository root:

```bash
python3.13 -m venv .venv-phase3
source .venv-phase3/bin/activate
python -m pip install -r phase3/requirements-test.txt
python -m uvicorn phase3.backend.api:app --host 127.0.0.1 --port 8002
```

In another shell with the environment activated:

```bash
DELIVERY_API_URL=http://127.0.0.1:8002 python -m streamlit run phase3/frontend/app.py
```

Open the printed Streamlit address (normally localhost:8501). The API exposes
Swagger at http://127.0.0.1:8002/docs and health at http://127.0.0.1:8002/health.
Inference uses packaged fitted artifacts and needs no raw training data.
For separate installs use `requirements-api.txt` and `requirements-ui.txt` with
the corresponding `constraints-api.txt` or `constraints-ui.txt`. The combined
test environment can resolve both; it also includes HTTP test support.

## Use the application

1. Open **Daily simulation**. Imagine the displayed current date is today.
   Dates range from 2 June to 1 July 2018, scoring the previous day's approvals.
2. Move with the date picker or **Previous day / Next day**. The cutoff is
   midnight at the start of the simulated date in the source's local wall time.
3. Set the follow-up proportion (30% initially) and minimum days remaining
   (1 initially). These are demonstration assumptions, not validated policies.
4. Inspect totals, eligible/excluded counts and reasons. Follow up the selected
   rows, ranked by C3 risk. R1 is supporting information, not a second filter. All eligible orders are scored;
   the proportion selects the top X% by risk for human follow-up.
5. Inspect an order or download the worklist. **Not selected** does not mean
   guaranteed on-time delivery.
6. Reveal retrospective outcomes explicitly to inspect historical selected late
   rate and coverage. Outcomes are hidden initially and reset on date/policy changes.
7. Use **Single-order API demo** to submit a sample or edit its engineered inputs.
   This calls the same services as the worklist. Batch CSV scoring is also available.

## Organisation

| Folder | Purpose |
|---|---|
| `backend/` | Independent fitted-model services, strict input validation and inference routes |
| `frontend/` | Streamlit UI, server-side historical orchestration, daily policy and packaged demo data |
| `artifacts/` | Fitted pipelines, metadata and packaging provenance; no historical order rows |
| `export/` | Offline export/packaging utilities; never run at web startup |
| `tests/` | Model, API, temporal policy, packaging and Streamlit checks |
| `docs/` | Deployment, model contracts, verification and archived prototype guide |

See [model and simulation contracts](docs/TECHNICAL.md),
[hosting instructions](docs/DEPLOYMENT.md) and [verification record](docs/VERIFICATION.md).

## Tests

```bash
python -m unittest discover -s phase3/tests -t . -v
python -m unittest discover -s phase2 -p 'test_*.py'
```

Some packaging tests compare against original Git history; run them in a full
checkout. Backend and frontend deployment images deliberately exclude the raw
source data, Phase 2 modules, offline training utilities and tests.

## Deployment status

The Render API is live and verified at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)). Both model endpoints passed HTTPS checks on 20 October 2026; the local Streamlit
worklist was verified against that HTTPS inference API. Streamlit public
deployment is pending at Andrew’s request; configure it with this API URL when ready.
See the verification record for checks actually performed.

For teammate connection details and a complete test request, see
[TEAMMATE_CONNECTION.md](docs/TEAMMATE_CONNECTION.md).

## API boundary

Render serves `/health`, `/model-info`, `/predict`, `/predict/batch`, independent
model routes and aggregate monitoring. Historical dates, orders, outcomes and
follow-up policy live only in Streamlit’s Python layer. The old demo endpoints
and `/explain` have been removed. Streamlit does not load models or reproduce
fitted preprocessing.
