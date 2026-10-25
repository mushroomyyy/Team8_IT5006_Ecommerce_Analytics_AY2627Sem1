# Deploy the reorganized repository

The existing Render API stays at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)). The previous Delivery
Watch app URL is
https://team8it5006ecommerceanalyticsay2627sem1-77gyui7xuhhsalu3sxrslc.streamlit.app/;
The migration is now in `main`; Jordan must replace it and the Phase 1 dashboard.
New URLs are pending. See [Streamlit handoff](STREAMLIT_HANDOFF.md).

## Published release — 25 October 2026

Andrew approved publication. Merge commit `456262a` is on remote `main`, and Render
deploy `dep-dbeto2g473hc739eqrig` is live after a 1m21s build/deployment. The service's
Dockerfile path is `./deployment/Dockerfile.api`. Public HTTP checks, unchanged
model versions, hosted/local prediction agreement and the 10/10 batch example pass.
See [live evidence](reorg_live_http_verification.json) and
[Render screenshot](screenshots/reorg-render-live.png). The working branch was
never pushed. The Streamlit handoff below is still pending.

## Release gate

`codex/repo-reorg` is a local-only working branch. Do not push it or create a PR
from it. Finish the checks in [REORG_VERIFICATION.md](REORG_VERIFICATION.md), then
Andrew reviews and approves the result. Only after that decision, merge the local
branch into `main` and push `main`. Jordan's `model-deployment-r1` branch remains
untouched; do not merge, delete or rebase it.

Coordinate the merge with Jordan because both old Streamlit entry points disappear
when remote `main` changes. The new app paths require replacement applications.

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

Before publishing `main`, inspect the service in Render. For a manually configured
service, update Settings → Build & Deploy → Dockerfile Path to the new path at
release time. For a Blueprint-managed service, confirm the `render.yaml` change
is applied. Keep the branch, URL, service name and remaining settings unchanged.
Deploy the newly approved `main` commit; do not add `[skip render]` to this merge.
Verify the live commit matches it and wait for `/health` to become healthy.

The API image installs `deployment/requirements-api.txt` with
`deployment/constraints-api.txt`. It copies only `deployment/api/`, its package
initializer, and the four runtime model/metadata files. It never trains models.
The Dockerfile-specific `deployment/Dockerfile.api.dockerignore` excludes the app
and its data; the root ignore file also supports the UI build, which needs them.

After deployment:

```bash
python -m deployment.scripts.verify_http https://team8-delivery-api.onrender.com
```

Confirm both model versions are unchanged, Swagger loads, and documented single
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

Jordan creates both replacement apps from `main`; the full configuration and
smoke checks are in [STREAMLIT_HANDOFF.md](STREAMLIT_HANDOFF.md). Delivery Watch
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
- Packaging equivalence tests need full Git history. They read the original R1
  artifact at fixed commit `bfea55986f266b4208a8c58f932bf67666391b92`, so moving
  Jordan's branch cannot silently change the comparison input.
