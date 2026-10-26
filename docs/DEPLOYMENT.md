# Deploy the reorganized repository

The existing Render API stays at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)). Both replacement
Streamlit apps were deployed from `main` at commit
`453076cf41a27e92c9850ae37f109283b09e9196` and verified on 26 October 2026:

| App | URL | Main file | Python |
|---|---|---|---|
| Delivery Watch | https://team8-delivery-watch.streamlit.app/ | `deployment/app/app.py` | 3.14 |
| Phase 1 dashboard | https://team8-olist-dashboard.streamlit.app/ | `deployment/phase1_dashboard/streamlit_app.py` | 3.14 |

Python versions match the previous apps' settings; the model API and local
artifact checks retain Python 3.13.5. Both old Streamlit apps were deleted only
after their replacements passed hosted checks. See [verification](VERIFICATION.md).

## Published release — 25 October 2026

Andrew approved publication. Merge commit `456262a` is on remote `main`, and Render
deploy `dep-dbeto2g473hc739eqrig` is live after a 1m21s build/deployment. The service's
Dockerfile path is `./deployment/Dockerfile.api`. Public HTTP checks, unchanged
model versions, hosted/local prediction agreement and the 10/10 batch example pass.
Detailed live verification records and screenshots are retained locally. The working branch was
never pushed. At this release, the Streamlit replacements were pending; they
were completed on 26 October 2026 as recorded above.

## Historical migration release gate

`codex/repo-reorg` is a local-only working branch. Do not push it or create a PR
from it. Finish the local verification checks, then
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

Both live apps deploy from `main` using the paths and runtimes listed above.
Delivery Watch uses the adjacent `deployment/app/requirements.txt` and the secret:

```toml
DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
```

Phase 1 uses `deployment/phase1_dashboard/requirements.txt` and no secrets.
Its launcher sets `OLIST_HOSTED_RELEASE=1` and extracts `data/olist_csv.zip`
into a temporary cache. First load can take a minute. The hosted dashboard
loaded successfully, its filters updated the metrics and its charts rendered.

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
