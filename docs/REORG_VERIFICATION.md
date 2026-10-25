# Repository reorganization verification — 25 October 2026

All checks below are local, using Python 3.13.5 on Andrew's Mac. The starting
commit is `be93fd24418b92a837395ecf01765e9494da7c3a`. Work stays on local-only
`codex/repo-reorg`. Remote `main`, Render and both hosted Streamlit apps have not
been changed. Publication and redeployment require Andrew's review decision.

## Tests

Before moving files:

| Suite | Result |
|---|---|
| Deployment (`phase3/tests` at the starting commit) | 76 passed; 56 subtests |
| Phase 2 (from its original working directory, excluding its archive) | 89 passed; 8 subtests |

After the move, `python -m pytest -q` discovers only `tests/` through `pytest.ini`:

```text
165 passed, 2 warnings, 67 subtests passed in 28.17s
```

The counts remain 76 deployment and 89 modelling tests. The three additional
subtests check the documented batch, synthetic and sample inputs against the
original regression artifact. Existing warnings concern Starlette's HTTP test
client and a deliberately unknown categorical value. Neither is a failure.

## Scientific and file preservation

[Preservation evidence](reorg_preservation_verification.json) verifies:

- All 219 files under `results/` are byte-identical to their originals.
- All 115 inventoried archive files and 11 packaged app-data files are unchanged.
- Of 24 modelling source files, only `src/data.py` changes: repository-root and
  bundled-ZIP paths. The bundled ZIP is byte-identical to the original Git file.
- Every existing notebook cell's outputs, execution count and metadata are
  unchanged. Each active notebook gains one root-setup cell; Phase 1 additionally
  updates its ZIP path. Every code cell passes syntax compilation.

[R1 equivalence evidence](reorg_model_equivalence.json) compares the pre-move
runtime artifact with the new artifact: exact predictions on 6,164 June replay
orders, 10 documented batch rows, one synthetic example and 30 sample fixtures.
The fitted coefficients, intercept, singular values, rank and metadata are equal.
No fitting runs. R1 now refers to `deployment.api.transforms.ExpandColumns`.
C3's SHA256 remains
`17577aa8c96c0f456b55ab8f7417bc5f2916d7f698e5a15c35926743ceb42682`.
Both model versions are unchanged.

The original R1 source is pinned to immutable commit
`bfea55986f266b4208a8c58f932bf67666391b92` in packaging and its equivalence test.
Jordan's branch is unchanged; no tag or remote branch is created. Original source
paths inside fitted-model metadata and the matching stats JSON remain historical
provenance, preserving C3's bytes and metadata equality. Export commands and
active file lookups use current paths.

## Notebooks and exports

[Notebook checks](reorg_notebook_verification.json) verify setup from both the
repository root and `notebooks/` for all five active notebooks. Their setup,
imports and initial data cells execute in temporary workspaces:

| Notebook | Check |
|---|---|
| Phase 1 EDA | Loads all nine raw tables; 99,441 raw orders |
| 01 data/features | Loads data, builds features and exports the initial inventory to scratch results |
| 02 classification | Builds the modelling population; 45,972 training rows; leakage/split assertions pass |
| 03 regression | Builds the modelling population; 45,371 training rows; leakage/split assertions pass |
| 04 summary | Reads both tasks' preserved result exports |

These are smoke checks, not full notebook executions. Training cells and figure
regeneration were not run. Saved outputs were never re-executed or replaced.
Archived notebooks were moved intact and are not claimed to run at their new paths.

The updated `deployment.export.export_eda` exporter ran from repository-root
imports with output and manifest redirected to a temporary directory. Its four
files have exactly the recorded hashes: `orders.parquet`, `seller_history.parquet`,
`repeat_monthly.parquet` and `data_checks.json`.
See [export evidence](reorg_eda_export_verification.json).

## Containers and API

Both Dockerfiles build successfully as linux/amd64 images on this ARM Mac:

```bash
docker build -f deployment/Dockerfile.api -t team8-api-reorg .
docker build -f deployment/Dockerfile.ui -t team8-ui-reorg .
```

The API container ran on local port 18026. Inspection confirms its image excludes
`deployment/app/`, `src/`, raw data and notebooks. It loads R1 through the new
package. The root `.dockerignore` supports the UI build; a Dockerfile-specific
API ignore file additionally removes the app/data tree from the API build context.

```bash
python -m deployment.scripts.verify_http http://127.0.0.1:18026
```

[HTTP evidence](reorg_local_http_verification.json) records healthy C3/R1, unchanged
model versions, Swagger, single/batch and independent/integrated numerical
agreement, invalid-request 422s and removed-route 404s. API version remains 2.1.0.
This is local verification, not a claim that the new layout is live on Render.

## Streamlit applications

[Application checks](reorg_app_verification.json) exercise all seven pages through
Streamlit AppTest with real HTTP calls to the local API, without mocked inference:

- Default 2 July simulation: 129 eligible/scored orders; 39 selected. Outcome
  reveal and next-day navigation also pass.
- Month-end review renders its July default.
- Explore: 98,958 orders; filtering to SP reduces this to 41,553.
- Single-order prediction succeeds; the batch preparation/scoring path returns
  10/10 successful predictions for `docs/examples/batch_example.csv`.
- Start here and Models and guidance render without errors.

The built UI container also runs on local port 18526, pointing at the API
container. A browser session confirms API connectivity and the 129/39 worklist.
[Browser screenshot](screenshots/reorg-local-daily.png).

[Phase 1 checks](reorg_phase1_verification.json) copy only its two Python entry
files and bundled ZIP into an isolated layout with no local CSV directory or
pre-existing extraction cache. The hosted wrapper extracts all nine tables,
loads 98,676 orders under its original purchase-date definitions, renders 26
Plotly charts and filters to 41,621 SP orders without errors. This was tested in
the local full environment; the hosted app retains its existing separate
requirements, including its pandas version range. Its original definition differs
from Explore's approval-date definition; neither was changed.

## Documentation and release status

Active source/configuration and current Markdown guides have no old deployment
package references. Historical documents under `docs/archive/`, archived notebooks
and original model/result provenance retain historical paths intentionally.
Active relative documentation links and whitespace checks pass. The devcontainer
now starts `deployment/app/app.py` with Python 3.13. The full local requirements
include the Phase 1 notebook plotting packages and pytest; hosting requirements
stay separate.

Pending after Andrew approves the review:

1. Coordinate a release window with Jordan for both replacement Streamlit apps.
2. Inspect/update the existing Render service's Dockerfile path at release time.
3. Merge the local branch into `main` and push `main`; never push the working branch.
4. Deploy the existing Render service and run live HTTP/model-version checks.
5. Jordan redeploys both Streamlit apps using [STREAMLIT_HANDOFF.md](STREAMLIT_HANDOFF.md),
   checks them and deletes the old apps after replacements pass.
6. Record both new URLs in active documentation with a link-only `[skip render]`
   follow-up commit, and update the report/slides outside this repository.

Do not interpret preserved historical deployment records as evidence that the
reorganized layout has already been published.
