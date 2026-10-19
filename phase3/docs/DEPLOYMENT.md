# Deploy API first, then Streamlit

No verified public URLs are recorded yet. Do not substitute a teaching-example
URL for an actual deployment. Deploy only the `classification-deployment` branch.

## Render FastAPI service

`render.yaml` defines a Docker web service. In the Render dashboard, connect
this GitHub repository and create a Blueprint from that file, or configure a
web service manually:

| Setting | Value |
|---|---|
| Branch | `classification-deployment` |
| Runtime | Docker |
| Dockerfile | `phase3/Dockerfile.api` |
| Build context | repository root |
| Health path | `/health` |
| Environment | `PORT=10000` (assigned Render port is honoured) |

The image installs pinned compatible dependencies, copies only runtime code and
fitted/demo artifacts, and starts `uvicorn phase3.backend.api:app` on `0.0.0.0:$PORT`.
It does not train models or require raw data. The blueprint requests a free plan;
check account availability before choosing any paid alternative.

A native Python service can instead use Python 3.13.5, build:

```bash
pip install -r phase3/requirements-api.txt -c phase3/constraints-api.txt
```

and start:

```bash
python -m uvicorn phase3.backend.api:app --host 0.0.0.0 --port "$PORT"
```

After deployment, use the actual HTTPS URL to verify `/health`, `/docs`, single
and batch prediction, date bounds and `/worklist`. Check cold startup from a
sleeping state and observe logs/memory. A localhost success is not live verification.

## Streamlit Community Cloud

After Render is verified, create an app for the same repository/branch. Set main
file `phase3/frontend/app.py` and Python 3.13. Select the frontend dependency file
located beside it (`phase3/frontend/requirements.txt`) to avoid installing backend
models and Phase 2 notebook dependencies. Streamlit normally discovers this file
beside the entry point. Set the app secret:

```toml
DELIVERY_API_URL = "https://YOUR-VERIFIED-RENDER-SERVICE.onrender.com"
```

The app also reads this environment variable locally. `CLASSIFICATION_API_URL`
is a compatibility fallback. Neither URL is a secret, but credentials must never
be committed. Example settings are in `frontend/.streamlit/secrets.toml.example`.

Verify both pages against the deployed API: navigation, prediction, downloads,
outcome reveal/reset, invalid input and timeout feedback. The frontend gives a
60-second request timeout and startup guidance for a sleeping backend.

A Docker frontend alternative is available as `phase3/Dockerfile.ui` with the
same API URL environment variable and assigned `PORT`.

## Local deployment-equivalent containers

From repository root:

```bash
docker build --platform linux/amd64 -f phase3/Dockerfile.api -t team8-phase3-api .
docker run --rm -e PORT=10000 -p 18002:10000 team8-phase3-api
docker build --platform linux/amd64 -f phase3/Dockerfile.ui -t team8-phase3-ui .
docker run --rm -e PORT=10000 -e DELIVERY_API_URL=http://host.docker.internal:18002 -p 18501:10000 team8-phase3-ui
```

## Troubleshooting

- `/health` returns 503: inspect model loading logs, correct artifact paths and
  pinned scikit-learn versions; deploy committed artifacts rather than refitting.
- Connection timeout: verify backend URL, HTTPS, service state and cold-start time.
- 422 input errors: use `/model-info` trained categories and the exact feature
  schema. Numeric blanks are supported, categorical blanks are not.
- Dates rejected: the frozen models are available from 2 June 2018; the final
  supported current date is 1 July for the 30 June approval batch.
- Missing outcomes: unknown retrospective labels are intentional and excluded
  from rate denominators, not assigned an on-time outcome.
- Streamlit imports backend libraries: verify the entry point and its adjacent
  requirements file; the Phase 3 UI imports only frontend dependencies.

Account blockers and checks actually performed are tracked in
[VERIFICATION.md](VERIFICATION.md).
