# Deploy API first, then Streamlit

The API is live and verified at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)), deployed on 20 October
2026 to a free Singapore web service. Public Streamlit deployment remains pending
at Andrew’s request. Deploy the `main` branch.

## Current branch and deployment flow

`main` is the deployment source for both Render and the new Phase 3 Streamlit
app. The integrated application is squash-merged into `main`, keeping the
implementation branch available for reference without carrying its intermediate
commit messages into main history.

Configure the live Render service to branch **`main`**, auto-deploy **On Commit**,
Dockerfile `./phase3/Dockerfile.api`, build context `.`, and health check `/health`.
The checked-in `render.yaml` specifies the same settings. For an existing manually
configured service, changing this file alone does not change its dashboard branch.

A normal push to `main` starts a Docker build and deploy. Render checks `/health`
before the new version becomes live. A commit containing `[skip render]` does not
trigger an auto-deploy; documentation/frontend-only changes can use this when no
API runtime changes need deployment. Runtime/model changes must deploy and be
verified normally. Confirm the dashboard's live commit matches the intended main
commit after changing the branch or deploying.

## Render FastAPI service

`render.yaml` defines a Docker web service. In the Render dashboard, connect
this GitHub repository and create a Blueprint from that file, or configure a
web service manually:

| Setting | Value |
|---|---|
| Branch | `main` |
| Runtime | Docker |
| Dockerfile | `phase3/Dockerfile.api` |
| Build context | repository root |
| Health path | `/health` |
| Environment | `PORT=10000` (assigned Render port is honoured) |

The image installs pinned compatible dependencies, copies only runtime code and
fitted model artifacts and metadata, and starts `uvicorn phase3.backend.api:app` on `0.0.0.0:$PORT`.
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
and batch prediction and the absence of the removed demo/explanation endpoints. Verify date boundaries and the daily
worklist through Streamlit’s Python layer against the live `/predict/batch` service. Check cold startup from a
sleeping state and observe logs/memory. A localhost success is not live verification.

## Streamlit Community Cloud

After Render is verified, create a separate Phase 3 app for the same repository
on branch `main`. Keep the existing Phase 1 app and its entry point unchanged. Set main
file `phase3/frontend/app.py` and Python 3.13. Select the frontend dependency file
located beside it (`phase3/frontend/requirements.txt`) to avoid installing backend
models and Phase 2 notebook dependencies. Streamlit normally discovers this file
beside the entry point. Set the app secret:

```toml
DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
```

The app also reads this environment variable locally. Its Python layer loads
`phase3/frontend/data/`, filters yesterday’s batch as of the cutoff, calls
`/predict/batch` for every eligible row, and ranks the returned risks. No model
artifacts or fitted preprocessing are included in the frontend image. `CLASSIFICATION_API_URL`
is a compatibility fallback. Neither URL is a secret, but credentials must never
be committed. Example settings are in `frontend/.streamlit/secrets.toml.example`.

Verify both pages against the deployed API: navigation, prediction, downloads,
outcome reveal/reset, invalid input and timeout feedback. The frontend gives a
10-second connection / 120-second read timeout and startup guidance for a sleeping backend.

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

## Streamlit publishing permissions

Streamlit Community Cloud requires GitHub repository **admin** permission to
deploy. If Andrew has only collaborator/write permission, Jordan must deploy
from this repository or explicitly grant admin access. Repository write access
alone does not establish publishing permission. See [Streamlit’s official account
connection guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account).

For documentation/evidence-only commits, `[skip render]` in the commit message
avoids restarting the already verified API; runtime/model changes must deploy and
be verified normally. See [Render deploy documentation](https://render.com/docs/deploys#skipping-an-auto-deploy).
