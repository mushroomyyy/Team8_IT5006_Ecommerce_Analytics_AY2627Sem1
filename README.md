# IT5006 E-Commerce Analytics — AY2026/27 Semester 1

Group 8 analyses delivery performance for approved Olist e-commerce orders: which
orders are likely to arrive late, and by how many days relative to the promise.
The project runs from exploratory analysis (Phase 1) through modelling (Phase 2)
to a deployed operations application (Phase 3).

| Live | URL |
|---|---|
| Delivery Watch | https://team8-delivery-watch.streamlit.app/ |
| Phase 1 dashboard | https://team8-olist-dashboard.streamlit.app/ |
| Model API (FastAPI on Render) | https://team8-delivery-api.onrender.com |
| API docs (Swagger, with examples) | https://team8-delivery-api.onrender.com/docs |

The free API sleeps when idle; the first request can take 30–60 seconds to wake it.
The C2B/R2B release (`cfd8f41`) was deployed and verified on 27 October 2026.
Render serves the new model versions, and Delivery Watch connects to them.
See the [live verification](docs/render_c2b_r2b_verification.json).
The directory migration is merged into `main`; Render was redeployed and verified
on 25 October 2026. Both replacement Streamlit apps were deployed and verified on
26 October 2026, using Python 3.14 and the migrated entry points. The old apps
have been deleted. See [deployment settings](docs/DEPLOYMENT.md) and the
[verification record](docs/VERIFICATION.md). The API service and URL are unchanged.

## Selected models

Models are chosen on the Validation set only: the Validation leader, a day-level bootstrap
of every model's difference from it, and the simplest model statistically tied with it.
Cross-validation only tunes and selects features within each model family, and June is an
independent test. Canceled and unavailable orders are excluded from training and evaluation
because they have no delivery outcome.

- **C2B logistic regression (13 BIC-selected features)**: late-delivery risk, used to rank orders
  for follow-up; operating cut-off 0.128 chosen on Validation.
- **R2B OLS regression (17 BIC-selected features)**: signed days from the promised delivery date
  (approval-to-delivery capped at 45 days).

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
| `deployment/export/`, `deployment/scripts/` | Offline model and data builders, scoring and verification commands |
| `deployment/phase1_dashboard/` | Original Phase 1 dashboard and hosted `streamlit_app.py` launcher |
| `deployment/Dockerfile.*`, `render.yaml` | Container builds and Render configuration |
| `docs/` | Application, API, modelling, deployment and verification guides |
| `archive/phase2_archive/` | Earlier exploratory work, preserved without edits |
| `requirements.txt`, `requirements-modelling.txt` | Full local environment and modelling dependencies |

## Phase 1: exploratory analysis

The original EDA notebook is [notebooks/phase1_eda.ipynb](notebooks/phase1_eda.ipynb).
Its saved charts and tables remain inside the notebook.

The original interactive dashboard lives in
[deployment/phase1_dashboard/](deployment/phase1_dashboard/README.md):

- `app.py`: dashboard implementation and charts.
- `streamlit_app.py`: Streamlit Cloud entry point; enables hosted mode and runs the dashboard.
- `requirements.txt`: the dashboard's separate hosting dependencies.

Both the notebook and dashboard use the raw tables bundled in `data/olist_csv.zip`.
The dashboard generates its charts when it runs. Delivery Watch's **Explore the
data** page is a separate implementation under `deployment/app/`, using Phase 2
definitions and small precomputed tables in `deployment/app/data/eda/`.

## Phase 2: modelling

The current modelling workflow consists of four notebooks, run in this order:

| Notebook | Purpose |
|---|---|
| [01_data_features_multicollinearity.ipynb](notebooks/01_data_features_multicollinearity.ipynb) | Load and audit data, build features, examine multicollinearity, and define chronological splits |
| [02_classification.ipynb](notebooks/02_classification.ipynb) | Fit and evaluate classification models |
| [03_regression.ipynb](notebooks/03_regression.ipynb) | Fit and evaluate regression models |
| [04_summary.ipynb](notebooks/04_summary.ipynb) | Read exported results and produce the combined summary |

These notebooks import the shared Python package in `src/`. Data loading,
features and labels live in `data.py`, `features.py` and `labels.py`; the remaining
modules provide preprocessing, variable selection, model fitting, decision-tree and
Random Forest tuning (`tree_search.py`, `rf_search.py`), Validation model choice
(`model_selection.py`), evaluation and plotting. Dependencies are in `requirements-modelling.txt`,
included by the root `requirements.txt`. Modelling tests are in `tests/phase2/`.

Phase 2's preserved exports are organised as follows:

```text
results/
├── feature_selection/                 # Multicollinearity and feature-selection tables
└── simple_to_complex/
    ├── data/                          # Data and feature audit tables from notebook 01
    ├── classification/                # Metrics, coefficients, tuning caches and metadata
    ├── regression/                    # Metrics, coefficients, tuning caches and metadata
    ├── figures/                       # Exported analysis and model charts
    ├── RESULTS_SUMMARY.md             # Combined results summary
    └── run_metadata.json              # Combined run information
```

Start with [RESULTS_SUMMARY.md](results/simple_to_complex/RESULTS_SUMMARY.md) to
read the findings, or [docs/MODELLING.md](docs/MODELLING.md) for the protocol and
source-module guide. Earlier notebooks and snapshots are preserved in
`archive/phase2_archive/`; they retain their historical paths and are not part of
the current execution workflow. A supporting plotting notebook also remains in
`results/simple_to_complex/figures/` as part of the preserved results.

## From modelling to deployment

The selected models are refitted on the June history and packaged as
`deployment/artifacts/classification_model.joblib` and `regression_model.joblib`;
the model code (C2B, R2B) is recorded inside each artifact. The exporters in
`deployment/export/` read the Validation choice from the notebooks'
`run_metadata.json`, reproduce the notebook's June scores and fitted-state
fingerprints exactly, and stop if they differ. The FastAPI service in
`deployment/api/` loads these artifacts for inference, and the Streamlit
application in `deployment/app/` calls that API over HTTP. Offline builders are
not run when the API or application starts.

## Setup

Python 3.13.5 matches the model API and fitted artifacts. The hosted Streamlit
apps use Python 3.14, matching their previous app settings.

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
on a follow-up list. Models were chosen on Validation; June is their held-out test
month and July and August are unseen monitoring months. Its seven pages cover getting started, daily simulation,
month-end review, data exploration, single-order and batch API demos, and models.

- [Application guide](docs/APPLICATION.md)
- [Calling the API](docs/API.md) · [Model and simulation contracts](docs/TECHNICAL.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Verification history](docs/VERIFICATION.md)

## Tests

```bash
python -m pytest tests/deployment -q
python -m pytest tests/phase2 -q
```

Hosting uses the smaller dependency sets under `deployment/`. A normal push to
`main` redeploys Render; its Dockerfile setting must match the new path before
that push. Both migrated Streamlit entry points now have verified replacement apps.
The final report and presentation are maintained separately by the team.
