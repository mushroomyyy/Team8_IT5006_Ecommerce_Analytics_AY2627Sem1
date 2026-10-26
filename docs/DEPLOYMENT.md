# Deploy the reorganized repository

The existing Render API stays at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)). The previous Delivery
Watch app URL is
https://team8it5006ecommerceanalyticsay2627sem1-77gyui7xuhhsalu3sxrslc.streamlit.app/;
The migration is now in `main`; Jordan must replace it and the Phase 1 dashboard.
New URLs are pending. The detailed collaborator handoff is retained locally.

**Current change: model switch to C2B and R2B.** The deployed models are now C2B
(`phase2-c2b-logistic-2018-06-02`) and R2B (`phase2-r2b-ols-2018-06-02`), replacing
C3 and R1. Locally verified on 26-27 October 2026; **live Render re-verification is
pending** and is done after the team merges to `main` and Render redeploys.

## Previous release (C3/R1) — 25 October 2026

Andrew approved publication of the repository reorganization. Merge commit `456262a` is on remote `main`, and Render
deploy `dep-dbeto2g473hc739eqrig` is live after a 1m21s build/deployment. The service's
Dockerfile path is `./deployment/Dockerfile.api`. Public HTTP checks, the C3/R1
model versions then served, hosted/local prediction agreement and the 10/10 batch example pass.
Detailed live verification records and screenshots are retained locally. The working branch was
never pushed. The Streamlit handoff below is still pending.

## Release gate

`ta-feedback-rework` carries the model switch and is not yet merged. Finish the
local verification checks, then Andrew reviews and approves the result. Only after
that decision, merge into `main` and push `main`; a push without `[skip render]`
rebuilds the Render service. Jordan's `model-deployment-r1` branch remains
untouched; do not merge, delete or rebase it.

Coordinate with Jordan about the Streamlit replacement apps, which also build from `main`.

## Existing Render service

| Setting | Value |
|---|---|
| Service name | `team8-delivery-api` (keep the existing service) |
| Branch | `main` |
| Runtime | Docker |
| Dockerfile | `./deployment/Dockerfile.api` |
| Build context | repository root (`.`) |
| Health check | `/health` |
| Region / plan | Singapore / free (unchanged) |
| Environment | `PORT=10000` |

The service, branch (`main`), Dockerfile path, health check and environment
variables are unchanged by the model switch, so no Render settings need editing.
Deploy the newly approved `main` commit; do not add `[skip render]` to this merge.
Verify the live commit matches it and wait for `/health` to become healthy.

The API image installs `deployment/requirements-api.txt` with
`deployment/constraints-api.txt`. It copies only `deployment/api/`, its package
initializer, and the four runtime model/metadata files
(`classification_model.joblib`, `regression_model.joblib` and their
`*_model_stats.json`). It never trains models.
The Dockerfile-specific `deployment/Dockerfile.api.dockerignore` excludes the app
and its data; the root ignore file also supports the UI build, which needs them.

After deployment:

```bash
python -m deployment.scripts.verify_http https://team8-delivery-api.onrender.com
```

Confirm `/health` reports `phase2-c2b-logistic-2018-06-02` and `phase2-r2b-ols-2018-06-02`, `/model-info` exposes C2B's `threshold` and `threshold_policy`, Swagger loads, and documented single
and batch examples score successfully. Observe logs for import or model-loading
failures. Local checks do not establish live success.

Rollback: use Render's previous successful deployment, or revert the approved
merge on `main`. A Git revert must also restore the corresponding Dockerfile
setting. Coordinate with Jordan if Streamlit apps have already been replaced.

## Local checks

From repository root:

```bash
docker build -f deployment/Dockerfile.api -t team8-api .
docker run --rm --name team8-api-reorg -p 10000:10000 team8-api
# In another shell:
python -m deployment.scripts.verify_http http://127.0.0.1:10000
```

On 26-27 October 2026 a linux/amd64 build of this image served the C2B/R2B
versions on `/health`, and `verify_http` passed all checks against it.

Optional UI container:

```bash
docker build -f deployment/Dockerfile.ui -t team8-ui .
docker run --rm -p 18501:10000 -e DELIVERY_API_URL=http://host.docker.internal:10000 team8-ui
```

Without containers:

```bash
python -m uvicorn deployment.api.api:app --host 127.0.0.1 --port 8002
# In a second shell:
DELIVERY_API_URL=http://127.0.0.1:8002 python -m streamlit run deployment/app/app.py
```

The devcontainer opens and starts the Phase 3 app at `deployment/app/app.py` using
Python 3.13. Set `DELIVERY_API_URL` to the intended API. The separate Phase 1
launcher is `deployment/phase1_dashboard/streamlit_app.py`.

## Streamlit

Jordan creates both replacement apps from `main`; the detailed handoff and
smoke checks are retained locally. Delivery Watch
uses the adjacent `deployment/app/requirements.txt` and the secret:

```toml
DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
```

Phase 1 uses `deployment/phase1_dashboard/requirements.txt` and no secrets. Do not
claim the new apps are live until Jordan supplies the URLs and checks pass.

## Troubleshooting

- `/health` returns 503: check import paths, committed artifacts and pinned
  serialization dependencies; restore a known artifact rather than retraining.
- App cannot reach API: check the URL, HTTPS, service state and cold-start wait.
- 422 input errors: use `/model-info` categories and the exact feature schema.
- Missing raw data: notebooks and Phase 1 use `data/olist_csv.zip`; the model API
  does not need that ZIP or `src/`.
- `/health` shows unexpected model versions: confirm the deploy used the intended
  `main` commit; the versions are inside each `*_model_stats.json`.
