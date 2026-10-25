# IT5006 E-Commerce Analytics — AY2026/27 Semester 1

Group 8 analyses delivery performance for approved Olist e-commerce orders: which
orders are likely to arrive late, and by how many days relative to the promise.
The project runs from exploratory analysis (Phase 1) through modelling (Phase 2)
to a deployed operations application (Phase 3).

| Live | URL |
|---|---|
| Delivery Watch app (Streamlit) | https://team8it5006ecommerceanalyticsay2627sem1-77gyui7xuhhsalu3sxrslc.streamlit.app/ |
| Model API (FastAPI on Render) | https://team8-delivery-api.onrender.com |
| API docs (Swagger, with examples) | https://team8-delivery-api.onrender.com/docs |

The free API sleeps when idle; the first request can take 30–60 seconds to wake it.

## Selected models

- **C3 tuned Random Forest**: late-delivery risk, used to rank orders for follow-up.
- **R1 OLS**: signed days from the promised delivery date (approval-to-delivery capped at 45 days).

Fitted preprocessing travels with each model; the API serves the packaged Phase 2 artifacts.

## Repository map

The repository is organised by project phase. This is where each part of the
course's suggested structure lives:

| Suggested folder | In this repository |
|---|---|
| `data/` | Raw Olist tables: `streamlit_release/data/olist_csv.zip` (extracted locally to `Olist_CSV/`, which is not committed) · packaged app data: `phase3/frontend/data/` |
| `notebooks/` | `phase2/01_data_features_multicollinearity.ipynb` … `04_summary.ipynb`; Phase 1 EDA: `IT5006_Group_Project_EDA_v2_(Group_8).ipynb` |
| `src/` | `phase2/src/` (data loading, features, labels, evaluation) |
| `deployment/` | `phase3/backend/` (FastAPI), `phase3/frontend/` (Streamlit), `phase3/Dockerfile.api`, `render.yaml` |
| `docs/` | `phase3/docs/` (API guide, technical notes, deployment, verification) |
| `README.md` | This file, plus `phase1_eda_dashboard/README.md` and `phase3/README.md` |

| Folder | Contents |
|---|---|
| [`phase1_eda_dashboard/`](phase1_eda_dashboard/README.md) | Phase 1 exploratory dashboard |
| `streamlit_release/` | Phase 1 hosted entry point and bundled raw data |
| `phase2/` | Modelling notebooks, shared source, results tables and tests |
| [`phase3/`](phase3/README.md) | Deployed API and app, fitted models, scripts, tests and docs |

## Setup

Python 3.13 is used for the deployed services and fitted artifacts.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt      # full environment: notebooks, API, app, tests
```

Raw data for the notebooks: unzip `streamlit_release/data/olist_csv.zip` into
`Olist_CSV/` (Phase 2 also extracts it automatically when missing). The deployed
API and app do not need the raw data.

Run the app locally against the live API:

```bash
DELIVERY_API_URL=https://team8-delivery-api.onrender.com python -m streamlit run phase3/frontend/app.py
```

Or run the API locally too (`python -m uvicorn phase3.backend.api:app --port 8002`)
and point `DELIVERY_API_URL` at `http://127.0.0.1:8002`.

## Phase 3 application

Each simulated morning between 2 June and 30 August 2018, the app takes the
previous day's approved orders, applies a midnight cutoff and eligibility rules,
scores every eligible order through the API, and puts the riskiest share
(30% by default) on a follow-up list. June guided model selection; July and
August are unseen monitoring months. Pages, in sidebar order: Start here (how to
use the app), Daily simulation, Month-end review, Explore the data (the Phase 1
dashboard rebuilt on Phase 2 definitions, with model outputs), single-order and
batch API demos, and Models and guidance.

- [Application guide](phase3/README.md)
- [Calling the API, with examples](phase3/docs/API.md)
- [Model and simulation contracts](phase3/docs/TECHNICAL.md)
- [Deployment](phase3/docs/DEPLOYMENT.md) · [Verification record](phase3/docs/VERIFICATION.md)

## Tests

```bash
python -m unittest discover -s phase3/tests -t . -v
python -m unittest discover -s phase2 -p 'test_*.py'
```

Phase 3 also ships smaller pinned dependency sets for hosting
(`phase3/requirements-api.txt`, `phase3/requirements-ui.txt`). Pushing to `main`
redeploys both the Render API and the Streamlit app. The final report and
presentation are maintained separately by the team.
