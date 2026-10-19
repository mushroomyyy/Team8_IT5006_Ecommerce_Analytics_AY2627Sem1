# IT5006 E-Commerce Analytics — AY2026/27 Semester 1

Group 8 models delivery performance for approved Olist e-commerce orders. The
project contains an exploratory dashboard, a reproducible Phase 2 modelling
pipeline, and a regression deployment slice.

## Repository map

| Area | Purpose |
|---|---|
| `Olist_CSV/` | Local Olist source tables used by the modelling pipeline |
| `phase1_eda_dashboard/` | Phase 1 exploratory Streamlit dashboard |
| `phase2/` | Final modelling notebooks, reusable source modules, tests and report exports |
| `deployment/` | FastAPI backend, R1 model packaging, Streamlit operations UI and deployment guide |
| `streamlit_release/` | Streamlit hosting entry point |
| `docs/` | Repository structure and deployment notes |

The existing `phase2/` paths are intentional: notebooks and source modules
use them for relative imports and data resolution. See
[`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for details.

## Selected models

The Phase 2 report selects:

- **C3 tuned Random Forest** for classification: late-delivery risk.
- **R1 all-feature OLS** for regression: days before or after the promised date.

The current deployment branch implements the regression slice first. Its
regression-derived operational flag marks a prediction above zero days as
`follow_up`; it is not a replacement for the C3 classification probability.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Phase 2 tests:

```bash
python -m unittest discover phase2 -p 'test_*.py'
```

Build the regression artifact and run the local UI:

```bash
python -m deployment.train_regression
python -m streamlit run deployment/streamlit_app.py
```

Run the API and open Swagger UI:

```bash
python -m uvicorn deployment.api:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

For Render commands and model packaging details, see
[`deployment/README.md`](deployment/README.md).

## Phase 1

The Olist exploratory dashboard and its setup instructions are in
[`phase1_eda_dashboard/README.md`](phase1_eda_dashboard/README.md).
