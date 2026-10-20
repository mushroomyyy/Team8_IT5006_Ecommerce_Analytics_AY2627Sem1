# IT5006 E-Commerce Analytics — AY2026/27 Semester 1

Group 8 analyses delivery performance for approved Olist e-commerce orders.
The project includes exploratory analysis, reproducible modelling and an
integrated operations application built with FastAPI and Streamlit.

| Area | Contents |
|---|---|
| [`phase1_eda_dashboard/`](phase1_eda_dashboard/README.md) | Phase 1 exploratory dashboard |
| `streamlit_release/` | Phase 1 hosted entry point and bundled data |
| `phase2/` | Modelling notebooks, shared source, results and reproducibility |
| [`phase3/`](phase3/README.md) | Integrated API, Streamlit frontend, fitted models, tests and guides |
| `deployment/` | Historical location; current code lives in Phase 3 |

Selected models are **C3 tuned Random Forest** for late-delivery risk and **R1
all-feature OLS** for signed days from the promised date, using a capped delivery
time target. Fitted preprocessing travels with each model.

## Phase 3 operations application

**Every morning, operations receives a prioritised list of yesterday's approved
orders, supported by classification risk and regression estimates.**

The main Streamlit page simulates 2 June–1 July 2018. It applies eligibility as of
the displayed cutoff and ranks yesterday's approvals by C3 risk. A configurable
30% review proportion starts the demo; it is not a validated universal policy.
R1 supplies supporting delivery estimates. A separate single-order page
illustrates the same API. Historical outcomes remain hidden until explicitly revealed.

```bash
python3.13 -m venv .venv-phase3
source .venv-phase3/bin/activate
python -m pip install -r phase3/requirements-test.txt
python -m uvicorn phase3.backend.api:app --host 127.0.0.1 --port 8002
```

In another activated shell:

```bash
DELIVERY_API_URL=http://127.0.0.1:8002 python -m streamlit run phase3/frontend/app.py
```

Open the Streamlit address printed in the terminal. Local Swagger:
http://127.0.0.1:8002/docs. **Live API:** https://team8-delivery-api.onrender.com ([Swagger](https://team8-delivery-api.onrender.com/docs)). The Streamlit frontend is built and locally verified; public frontend deployment is pending.
See the [application guide](phase3/README.md), [model/simulation contracts](phase3/docs/TECHNICAL.md),
[hosting setup](phase3/docs/DEPLOYMENT.md), and [verification record](phase3/docs/VERIFICATION.md).

## Reproducibility and tests

Phase 2 notebooks and source paths remain intact. Raw Olist data is needed for
reproduction/export, but not Phase 3 inference. The bundled ZIP remains at
`streamlit_release/data/olist_csv.zip`.

```bash
python -m unittest discover -s phase3/tests -t . -v
python -m unittest discover -s phase2 -p 'test_*.py'
```

Install root `requirements.txt` for the full notebook/development environment;
Phase 3 provides smaller backend and frontend dependency sets for hosting.
The [Phase 1 instructions](phase1_eda_dashboard/README.md) remain the entry point
for the exploratory dashboard. The final report and presentation are maintained
separately by the team.
