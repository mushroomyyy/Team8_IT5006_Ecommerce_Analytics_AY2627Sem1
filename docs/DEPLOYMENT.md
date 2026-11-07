# Deployment

The project runs as three hosted services, all deployed from `main`:

| Service | URL | Entry point | Python |
|---|---|---|---|
| Model API (FastAPI, Render) | https://team8-delivery-api.onrender.com ([Swagger](https://team8-delivery-api.onrender.com/docs)) | `deployment/Dockerfile.api` | 3.13.5 |
| Delivery Watch (Streamlit) | https://team8-delivery-watch.streamlit.app/ | `deployment/app/app.py` | 3.14 |
| Phase 1 dashboard (Streamlit) | https://team8-olist-dashboard.streamlit.app/ | `deployment/phase1_dashboard/streamlit_app.py` | 3.14 |

The API serves the Validation-selected models C2B (`phase2-c2b-logistic-2018-06-02`)
and R2B (`phase2-r2b-ols-2018-06-02`). Live checks are recorded in
[verification](VERIFICATION.md) and
[render_c2b_r2b_verification.json](render_c2b_r2b_verification.json).

## Render (model API)

| Setting | Value |
|---|---|
| Service name | `team8-delivery-api` |
| Branch | `main` |
| Runtime | Docker |
| Dockerfile | `./deployment/Dockerfile.api` |
| Build context | repository root (`.`) |
| Health check | `/health` |
| Region / plan | Singapore / free |
| Environment | `PORT=10000` |

These settings are also declared in `render.yaml`. Each push to `main` triggers an
automatic deploy (commits marked `[skip render]` do not); wait for `/health` to
report healthy before checking the service.

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

Rollback: redeploy Render's previous successful deployment, or revert the commit
on `main`.

## Local checks

From repository root:

```bash
docker build -f deployment/Dockerfile.api -t team8-api .
docker run --rm --name team8-api -p 10000:10000 team8-api
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

Both apps deploy from `main` using the entry points and runtimes listed above.
Delivery Watch uses the adjacent `deployment/app/requirements.txt` and the secret:

```toml
DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
```

Phase 1 uses `deployment/phase1_dashboard/requirements.txt` and no secrets.
Its launcher sets `OLIST_HOSTED_RELEASE=1` and extracts `data/olist_csv.zip`
into a temporary cache, so the first load can take a minute.

## Troubleshooting

- `/health` returns 503: check import paths, committed artifacts and pinned
  serialization dependencies; restore a known artifact rather than retraining.
- App cannot reach API: check the URL, HTTPS, service state and cold-start wait.
- 422 input errors: use `/model-info` categories and the exact feature schema.
- Missing raw data: notebooks and Phase 1 use `data/olist_csv.zip`; the model API
  does not need that ZIP or `src/`.
- `/health` shows unexpected model versions: confirm the deploy used the intended
  `main` commit; the versions are inside each `*_model_stats.json`.
