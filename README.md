# IT5006 E-Commerce Analytics — AY2026/27 Semester 1

Group 8 analyses delivery performance for approved Olist e-commerce orders: which
orders are likely to arrive late, and by how many days relative to the promise.
The project runs from exploratory analysis (Phase 1) through modelling (Phase 2)
to a deployed operations application (Phase 3).

| Live | URL |
|---|---|
| Delivery Watch (previous URL; replacement pending) | https://team8it5006ecommerceanalyticsay2627sem1-77gyui7xuhhsalu3sxrslc.streamlit.app/ |
| Phase 1 dashboard | Replacement URL pending Jordan's deployment |
| Model API (FastAPI on Render) | https://team8-delivery-api.onrender.com |
| API docs (Swagger, with examples) | https://team8-delivery-api.onrender.com/docs |

The free API sleeps when idle; the first request can take 30–60 seconds to wake it.
The directory migration is merged into `main`; Render was redeployed and verified
on 25 October 2026. Jordan must now replace both Streamlit apps using the
[deployment handoff](docs/STREAMLIT_HANDOFF.md). Their old entry points have been
removed, so the previous URLs may no longer load. New URLs will be recorded after
deployment. The API service and URL are unchanged.

## Selected models

- **C3 tuned Random Forest**: late-delivery risk, used to rank orders for follow-up.
- **R1 OLS**: signed days from the promised delivery date (approval-to-delivery capped at 45 days).

Fitted preprocessing travels with each model; the API serves the packaged Phase 2 artifacts.

## Repository map

| Path | Contents |
|---|---|
| `data/olist_csv.zip` | Bundled raw Olist CSVs; extracted `Olist_CSV/` stays local |
| `notebooks/phase1_eda.ipynb` | Phase 1 exploratory notebook |
| `notebooks/01_*.ipynb` … `04_*.ipynb` | Phase 2 data/features, classification, regression and summary |
| `src/` | Phase 2 data loading, features, labels, modelling and evaluation |
| `results/` | Preserved Phase 2 tables and figures |
| `tests/phase2/`, `tests/deployment/` | Modelling and deployment tests |
| `deployment/api/` | FastAPI inference service |
| `deployment/app/` | Delivery Watch Streamlit app, adjacent requirements and packaged data |
| `deployment/artifacts/` | Fitted models, metadata and manifest |
| `deployment/export/`, `deployment/scripts/` | Offline builders, scoring and verification commands |
| `deployment/phase1_dashboard/` | Original Phase 1 dashboard and hosted `streamlit_app.py` launcher |
| `deployment/Dockerfile.*`, `render.yaml` | Container builds and Render configuration |
| `docs/` | Application, API, modelling, deployment and verification guides |
| `archive/phase2_archive/` | Earlier exploratory work, preserved without edits |
| `requirements.txt`, `requirements-modelling.txt` | Full local environment and modelling dependencies |

## Setup

Python 3.13.5 matches the deployed services and fitted artifacts.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebook loader extracts `data/olist_csv.zip` into `Olist_CSV/` when needed.
Launch Jupyter from the repository root; each active notebook also supports a
kernel started in `notebooks/` through its first setup cell. Saved notebook outputs
and `results/` are preserved evidence. See [modelling](docs/MODELLING.md).

Run the app locally against the live API:

```bash
DELIVERY_API_URL=https://team8-delivery-api.onrender.com python -m streamlit run deployment/app/app.py
```

Or start a local API and point the app at it:

```bash
python -m uvicorn deployment.api.api:app --host 127.0.0.1 --port 8002
# In a second shell, with the same environment activated:
DELIVERY_API_URL=http://127.0.0.1:8002 python -m streamlit run deployment/app/app.py
```

The Phase 1 hosted launcher uses the bundled archive:

```bash
python -m streamlit run deployment/phase1_dashboard/streamlit_app.py
```

## Application and guides

Each simulated morning between 2 June and 30 August 2018, the app scores the
previous day's eligible approvals and puts the riskiest share (30% by default)
on a follow-up list. June guided model selection; July and August are unseen
monitoring months. Its seven pages cover getting started, daily simulation,
month-end review, data exploration, single-order and batch API demos, and models.

- [Application guide](docs/APPLICATION.md)
- [Calling the API](docs/API.md) · [Model and simulation contracts](docs/TECHNICAL.md)
- [Deployment](docs/DEPLOYMENT.md) · [Streamlit handoff](docs/STREAMLIT_HANDOFF.md)
- [Verification history](docs/VERIFICATION.md) · [Reorganization checks](docs/REORG_VERIFICATION.md)

## Tests

```bash
python -m pytest tests/deployment -q
python -m pytest tests/phase2 -q
```

Hosting uses the smaller dependency sets under `deployment/`. A normal push to
`main` redeploys Render; its Dockerfile setting must match the new path before
that push. Both Streamlit entry points need replacement apps after the migration.
The final report and presentation are maintained separately by the team.
