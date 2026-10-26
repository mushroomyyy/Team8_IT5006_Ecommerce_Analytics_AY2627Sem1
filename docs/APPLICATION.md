# Phase 3: delivery operations

Every morning, operations receives a prioritised list of yesterday's approved
orders, supported by C2B classification risk and R2B regression estimates.
Streamlit’s server-side Python code loads the historical demo, applies eligibility
and ranks results returned by one FastAPI model-inference service over HTTP.

## Run locally

Use Python 3.13.5, matching the tested deployment images and fitted artifacts.
From the repository root:

```bash
python3.13 -m venv .venv-phase3
source .venv-deployment/bin/activate
python -m pip install -r deployment/requirements-test.txt
python -m uvicorn deployment.api.api:app --host 127.0.0.1 --port 8002
```

In another shell with the environment activated:

```bash
DELIVERY_API_URL=http://127.0.0.1:8002 python -m streamlit run deployment/app/app.py
```

Open the printed Streamlit address (normally localhost:8501). The API exposes
Swagger at http://127.0.0.1:8002/docs and health at http://127.0.0.1:8002/health.
Inference uses packaged fitted artifacts and needs no raw training data.
For separate installs use `requirements-api.txt` and `requirements-ui.txt` with
the corresponding `constraints-api.txt` or `constraints-ui.txt`. The combined
test environment can resolve both; it also includes HTTP test support.

## Use the application

1. **Start here** explains each page. Then open **Daily simulation**. Imagine the displayed current date is today.
   It opens on 2 July 2018 (the first unseen morning); dates range from 2 June to
   30 August 2018, scoring the previous day's approvals.
   The models were chosen on Validation: June is their held-out test month, and
   July and August are unseen monitoring months.
2. Move with the date picker or **Previous day / Next day**. The cutoff is
   midnight at the start of the simulated date in the source's local wall time.
3. Set the follow-up proportion (30% initially) and minimum days remaining
   (1 initially). These are demonstration assumptions, not validated policies.
4. Inspect totals, eligible/excluded counts and reasons. Follow up the selected
   rows, ranked by C2B risk. R2B is supporting information, not a second filter. All eligible orders are scored;
   the proportion selects the top X% by risk for human follow-up. Canceled orders
   stay on the list because cancellation is unknown at approval.
5. Inspect an order or download the worklist. **Not selected** does not mean
   guaranteed on-time delivery.
6. Reveal retrospective outcomes explicitly to inspect historical selected late
   rate and coverage; canceled orders show as "Canceled" and are excluded from both. Outcomes are hidden initially and reset on date/policy changes.
7. Open **Month-end review** and pick a month to see what the daily list would
   have caught over the whole month, compared with a whole-month ranking, random
   calls and a ranking by R2B's buffer, plus R2B's timing accuracy. This page uses
   pre-computed scores and works while the API is asleep.
8. Open **Explore the data** for the Phase 1 dashboard rebuilt on Phase 2
   definitions: filters by month, state, route and category (canceled and unavailable orders are not counted as late); tabs for orders,
   delivery, geography, sellers, customers (repeat purchases and reviews), model
   outputs and data definitions. Refresh its tables with
   `python -m deployment.export.export_eda`.
9. Use **Single-order API demo** to submit a sample or edit its engineered inputs.
   This calls the same services as the worklist and shows whether the order is
   Flagged or Not flagged at C2B's operating cut-off. Batch CSV scoring is also available;
   `docs/examples/batch_example.csv` is a ready-made file of 10 orders.
10. **Models and guidance** summarises both models, the Validation model-choice
   tables, C2B's confusion counts at 0.5 and at the operating cut-off, the June
   late orders it missed, month-by-month performance and live API monitoring from
   `/logs/summary`. Model labels come from the API metadata.

## Organisation

| Folder | Purpose |
|---|---|
| `deployment/api/` | Independent fitted-model services, strict input validation and inference routes |
| `deployment/app/` | Streamlit UI, server-side historical orchestration, daily policy and packaged demo data |
| `deployment/artifacts/` | Fitted pipelines (`classification_model.joblib`, `regression_model.joblib`), metadata and manifest; no historical order rows |
| `deployment/export/` | Offline training and export utilities; never run at web startup |
| `deployment/scripts/` | `score_replay.py` (pre-scores the replay for Month-end review), `evaluate_daily.py`, `verify_http.py` |
| `tests/deployment/` | Model, API, temporal policy, packaging and Streamlit checks |
| `docs/` | API guide, deployment, model contracts, verification and archived prototype guide |

See [calling the API](API.md), [model and simulation contracts](TECHNICAL.md),
[hosting instructions](DEPLOYMENT.md) and [verification record](VERIFICATION.md).

## Tests

```bash
python -m pytest tests/deployment -q
python -m pytest tests/phase2 -q
```

Backend and frontend deployment images deliberately exclude the raw
source data, Phase 2 modules, offline training utilities and tests.

## Deployment status

The repository migration was published on `main` as merge commit `456262a`, and
the Render API at https://team8-delivery-api.onrender.com
([Swagger](https://team8-delivery-api.onrender.com/docs)) was verified live on
25 October 2026 serving the previous C3/R1 models.

This branch switches the deployed models to C2B and R2B. It was verified locally on
26-27 October 2026 (Docker build served the new versions; `verify_http` passed
all checks; 180 tests pass). **Live Render re-verification is pending**: the team
redeploys after merging, so do not assume the live service serves C2B/R2B until
that check is recorded.

Both replacement Streamlit apps were deployed from `main` at commit
`453076cf41a27e92c9850ae37f109283b09e9196` and verified on 26 October 2026:

- [Delivery Watch](https://team8-delivery-watch.streamlit.app/): `deployment/app/app.py`.
- [Phase 1 dashboard](https://team8-olist-dashboard.streamlit.app/): `deployment/phase1_dashboard/streamlit_app.py`.

Both use Python 3.14, matching the previous apps' runtime settings. Delivery
Watch's seven pages and the 10/10 example batch passed hosted checks; Phase 1's
filters and charts also passed. The old apps were deleted after verification.
See [deployment settings](DEPLOYMENT.md) and the
[verification record](VERIFICATION.md). For API clients, see [API.md](API.md).

## Refreshing packaged data

Run from the repository root after changing the replay window or the models, in
this order:

```bash
python -m deployment.export.train_classification  # refit C2B from the notebook's selected_model
python -m deployment.export.train_regression      # refit R2B likewise
python -m deployment.export.export_simulation     # needs Olist_CSV/; writes orders, outcomes, timing targets, manifest
python -m deployment.scripts.score_replay         # pre-scores every replay order
python -m deployment.export.export_eda            # refreshes the Explore tables
python -m deployment.scripts.evaluate_daily       # writes docs/daily_policy_evaluation.json
```

## API boundary

Render serves `/health`, `/model-info`, `/predict`, `/predict/batch`, independent
model routes and aggregate monitoring. Historical dates, orders, outcomes and
follow-up policy live only in Streamlit’s Python layer. The old demo endpoints
and `/explain` have been removed. Streamlit does not load models or reproduce
fitted preprocessing.
