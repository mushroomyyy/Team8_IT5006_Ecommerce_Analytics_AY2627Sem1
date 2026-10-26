# Verification record

## Full notebook rerun — 27 October 2026

All four current notebooks ran sequentially, top to bottom in fresh kernels, in
375.2 seconds with cached Random Forest and decision-tree search results. No code
cell failed. The full suite passes 181 tests and 68 subtests.

- C2B and R2B remain selected. Their saved per-split and CV metrics reproduce
  exactly; selected feature sets, operating threshold and fitted-state fingerprints
  are unchanged. Deployment artifact files are byte-identical and need no rebuild.
- Default classifier C3d's July AP differs by 7.31e-8 and ROC-AUC by 3.19e-7.
  Other model-metric differences are at floating-point precision (regression at
  most 3.56e-15). Reported rounded headline results are unchanged.
- The reporting helper now uses the scoring eligibility rule for both tasks.
  Regression July is 6,078 scored/evaluated; August is 6,561 scored/evaluated.
  Notebook 01's table and timeline, including the image embedded in Notebook 02,
  have been regenerated.
- Notebook displays now group primary and supporting metrics, show F1 beside
  precision/recall at 0.5, and clarify percentage scales and shared thresholds.
  Narratives qualify retrospective June assessment and the parsimony heuristic;
  an interval containing zero does not establish equivalence.
- [Machine-readable rerun verification](notebook_rerun_verification.json).


Latest live checks: 27 October 2026 (Singapore). The C2B/R2B release is merged
and pushed to `main` as `cfd8f41`. Render deploy `dep-dbfnp8ajnfac73clqoeg`
succeeded in 56.4 seconds. All eight live HTTP check groups pass and the hosted
synthetic-example predictions match the local packaged models. All 180 local tests pass.

## C2B/R2B release verified live — 27 October 2026

- `/health` serves `phase2-c2b-logistic-2018-06-02` and
  `phase2-r2b-ols-2018-06-02`. `/model-info` exposes C2B's Validation-selected
  threshold, `0.12819778362486173`, and its policy.
- Live verification covers health, model metadata, single/batch agreement,
  independent/integrated agreement, invalid-input 422s, removed-route 404s,
  inference-only OpenAPI/metadata and Swagger. TLS verification remained enabled;
  the local Python invocation used the installed certifi CA bundle.
- Delivery Watch auto-updated at https://team8-delivery-watch.streamlit.app/.
  Its sidebar shows **API connected · C2B + R2B**. Models and guidance renders
  the 13-feature C2B and 17-feature R2B identities, model-selection tables,
  confusion metrics and false-negative analysis.
- Daily simulation scores all 129 eligible approvals for 2 July 2018 using the
  live API, with 39 selected at the default 30% follow-up proportion. Outcomes
  remain hidden initially. These two pages were checked after release; the
  other pages were not rechecked in this run.
- [Machine-readable live verification](render_c2b_r2b_verification.json).
- Render settings are unchanged: `main`, `deployment/Dockerfile.api`, `/health`,
  Singapore/free and `PORT=10000`. No rollback was needed.

The Streamlit verification dated 26 October 2026 and every entry dated 25 October 2026 or earlier describe the previous C3/R1 deployment (C3 tuned Random Forest, R1 all-feature OLS), including the `docs/render_*.json` and `docs/streamlit_live_api_verification.json` records, which are left unchanged.

## Model switch to C2B and R2B — 26-27 October 2026

Pre-release local checks; live verification is recorded above.

- Deployed models are now C2B (`phase2-c2b-logistic-2018-06-02`, 13 BIC-selected
  features, operating cut-off 0.1282 from Validation) and R2B
  (`phase2-r2b-ols-2018-06-02`, 17 BIC-selected features). They were chosen on
  Validation only; June is an independent held-out test and July/August are
  monitoring. Canceled and unavailable orders are excluded from training and
  evaluation.
- Artifacts are `classification_model.joblib` and `regression_model.joblib`.
  `classification_c3_rf.joblib`, `regression_r1_ols.joblib`,
  `regression_packaging.json` and `package_regression.py` were removed.
- The exporters refit on the June history and verified against the notebooks:
  June AP 0.08147 and June RMSE 7.3738 match the reference values exactly, and
  both fitted-state fingerprints equal the notebook fingerprints.
- A linux/amd64 Docker build served `phase2-c2b-logistic-2018-06-02` and
  `phase2-r2b-ols-2018-06-02` on `/health`, and
  `python -m deployment.scripts.verify_http http://localhost:18002` passed all checks.
- 180 tests pass.
- `docs/daily_policy_evaluation.json` was regenerated for C2B (30% review, one day
  remaining, rounding up): 6,164 June approvals, 6,134 known, 8 unknown, 116 known
  late; 1,863 selected, 1,855 known, 3 unknown, 78 known late. Selected late rate
  78/1,855 = 4.20%; eligible/all known late coverage 78/116 = 67.24%. The remaining
  22 approvals (5 of them selected) are canceled and have no outcome. June is held-out test data, but
  the policy settings are demonstration assumptions.
- C2B on June: average precision 0.0815, ROC-AUC 0.729; at the cut-off it flags
  6.3% of orders (34 of 116 known late caught); at 0.5 it flags none. R2B on June:
  RMSE 7.374, MAE 5.81 days. Monitoring ROC-AUC: July 0.591, August 0.560.
- Completed on 27 October 2026: release merge, Render redeploy, live HTTP checks
  and hosted Streamlit model-guidance/daily-simulation checks (see above).

## Streamlit replacements verified — 26 October 2026 (previous C3/R1 deployment)

- Owner: `mushroomyyy`. Both apps were created from repository `main` at source
  commit `453076cf41a27e92c9850ae37f109283b09e9196`.
- Delivery Watch: https://team8-delivery-watch.streamlit.app/ using
  `deployment/app/app.py`, its adjacent requirements and the Render API URL secret.
- Phase 1: https://team8-olist-dashboard.streamlit.app/ using
  `deployment/phase1_dashboard/streamlit_app.py`, its adjacent requirements and no secrets.
- Both hosted apps use Python 3.14, matching the runtime selected in each old
  app's settings. The local frontend checks used Python 3.13; this does not
  change the Python 3.13.5 model API runtime or fitted artifacts.
- All seven Delivery Watch pages passed hosted browser checks. Start here loads;
  the default 2 July 2018 daily list has 129 scored orders and 39 selected.
  Next day shows 162/49; previous day restores 129/39. Revealed outcomes reset
  on date changes. The worklist download contains 129 CSV rows.
- July month-end review shows 127 of 318 late deliveries caught. Explore defaults
  to 98,958 approved orders and renders 24 charts; advancing the starting month
  to February 2017 changes the count to 98,198, and restoring January returns
  98,958. No application exceptions appeared.
- Single-order prediction returns both C3 risk and R1 signed days from promise.
  Uploading `docs/examples/batch_example.csv` through the hosted app and scoring
  it produces a downloadable CSV with 10/10 successful rows and finite results
  from both models. Both model versions and guidance render correctly.
- Phase 1 loads the bundled data and renders its charts without an application
  exception or observed memory kill. The default filter shows 98,676 orders;
  selecting delivered orders gives 95,804, and restoring all orders gives 98,676.
- The public Delivery Watch root URL was checked in a fresh browser session.
  An incomplete initial background-tab render was resolved by opening a fresh
  session; the sidebar and navigation then rendered without a code change.
- The live Render HTTP verifier passes all eight check groups, with model
  versions `phase2-c3-rf-2018-06-02` and `phase2-r1-ols-2026-10-09` unchanged.
- Existing local UI and simulation checks: 39 tests and 14 subtests passed.
- Both old Streamlit apps were deleted after replacement verification. The
  Streamlit dashboard now lists only the two migrated entry points above.
- No outstanding failed deployment checks. Separately maintained slides/PDF,
  final report and previously shared links still need the team's new URLs.

## Reorganization published — 25 October 2026 (previous C3/R1 deployment)

- Render deploy `dep-dbeto2g473hc739eqrig` serves merge `456262a` using
  `./deployment/Dockerfile.api`; the dashboard reports Live after 1m21s.
- Live HTTPS checks pass: health, Swagger,
  inference metadata, single/batch and independent/integrated agreement,
  invalid-input 422s and removed-route 404s.
- C3/R1 model versions are unchanged; hosted predictions agree with the local
  models. The documented example batch succeeds 10/10.
- Detailed migration evidence and screenshots are retained locally.
- Only `main` was pushed. Both Streamlit replacement deployments remain pending;
  the detailed collaborator instructions are retained locally.

The dated sections below are historical verification records of the previous C3/R1 deployment; their model names, counts and results do not describe C2B/R2B.

## Streamlit / inference split — 20 October 2026 (C3/R1)

- Historical data, cutoff eligibility, ranking, follow-up policy and retrospective
  metrics now execute in Streamlit’s Python server. Render is inference-only.
- All eligible rows are scored before selecting the top configurable percentage
  for human follow-up. The setting is labelled **Follow-up proportion (%)**.
- `/samples`, `/simulation`, `/worklist`, `/outcomes` and `/explain` are removed
  from API 2.1.0; the sensitivity UI is removed too.
- **61 current Phase 3 tests pass**, including strict validation, numerical model
  equivalence, temporal policy, HTTP row-error mapping, cache behavior, outcome
  reveal without rescoring, CSV precision and API isolation from frontend/data.
- Both clean Docker images build. The API image copies only backend code and four
  model/metadata files. The frontend image has historical JSON and Python helpers,
  without fitted model artifacts.
- The fitted model bytes and fixed daily-policy retrospective results are unchanged.
- API 2.1.0 is live at https://team8-delivery-api.onrender.com from commit
  `f44090701a66c330bb6ff628353e67dbcddd601e` (20 October 2026).
  [Current HTTPS verification](render_inference_http_verification.json) passed:
  both models healthy, single/batch numerical equivalence, independent/integrated
  agreement, invalid single/empty-batch 422 errors, removed route 404 responses,
  inference-only OpenAPI/metadata and Swagger. Full sequence took 2.196 seconds; Render reported a 1m02s deployment.
- Local Streamlit container at http://127.0.0.1:18523 is connected directly to
  the live HTTPS model API. Firefox verified the worklist, explicit outcome reveal,
  date navigation/outcome reset and selection-count rounding. The first day has
  147 eligible/scored rows and 45 follow-up recommendations; second day 164/50; a fresh third day against API 2.1 has 156/47.
- The UI explains that the first connection after inactivity may take 50 seconds
  or more, shows the elapsed wait and supports retry. Health checks use 5/10 s
  timeouts within a 120 s wake budget; scoring uses 5/60 s per 100-order chunk.
- The Streamlit app is deployed on Streamlit Community Cloud from `main`. Anyone
  can call the API without a Render account; see [API.md](API.md).

## Initial integration checks (before the boundary refactor; C3/R1)

- Regression branch merged into `classification-deployment`; original dirty work
  backed up before integration and retained in moved sources/artifacts/archives.
- **52 Phase 3 tests pass**: actual-model contracts, single/batch equality, strict
  input validation, temporal eligibility, ties/rounding/small and empty batches,
  unknown-outcome denominators, date navigation, outcome reveal/reset, HTTP-only
  frontend, error states and downloaded CSV reupload.
- **89 Phase 2 tests pass**. Phase 2 modelling code/results are preserved.
- Independent review completed. Stale manifest and CSV result-column reupload
  issues were fixed and covered by passing checks.
- C3 binary exactly matches the pre-integration working copy. Original vs
  standalone R1 predictions match exactly on all **6,164** June replay orders.
  All 6,164 snapshot rows pass both model validators.
- Both `linux/amd64` Docker images install from clean `python:3.13.5-slim` bases.
  API image contains no Phase 2 source or raw data; UI image contains no models.
- Fresh API container tested at `http://127.0.0.1:18023`: health, Swagger,
  metadata, samples, integrated single/batch exact equality, both date boundaries,
  outcome separation, retrospective endpoint and invalid date/input 422 responses.
  Reusable check: `python -m deployment.scripts.verify_http ACTUAL_BASE_URL`.
- Fresh Streamlit container tested at `http://127.0.0.1:18523` against that API:
  daily worklist, next/back navigation, explicit outcomes/re-hide, selected-order
  transfer and real C3/R1 single-order prediction all verified in browser.
  Initial 2 June example: 147 eligible approvals, 45 selected. Selected-order
  prediction reproduced its worklist values (14.1% risk, -20.19 days from promise).
- Local container memory snapshot: API approximately 219 MiB; UI approximately
  64 MiB before browser sessions. These are local observations, not guaranteed
  hosting limits or live cold-start measurements.
- Phase 1 launcher restored to the exploratory dashboard; bundled source ZIP
  remains available. Existing Phase 1 requirements and application paths remain intact.

## Fixed daily-policy retrospective results (previous C3 deployment)

`python -m deployment.scripts.evaluate_daily` writes
[daily_policy_evaluation.json](daily_policy_evaluation.json), now regenerated for
C2B (see the model switch entry above); the table below is the earlier C3 result. Defaults are 30%
review and more than one calendar day remaining, rounding selection up each day.
No June outcomes were used to tune these preset operational settings.

| Population | Total | Known | Unknown | Known late |
|---|---:|---:|---:|---:|
| All June approvals | 6,164 | 6,155 | 9 | 137 |
| Eligible at daily cutoff | 6,164 | 6,155 | 9 | 137 |
| Selected across daily lists | 1,863 | 1,855 | 8 | 80 |

Default eligibility happens to exclude no orders in these packaged June batches;
exclusions are tested with controlled timestamps and configurable policy settings.
Daily rounding produces 1,863 selected orders, rather than the report's monthly
1,850. Selected-known late rate is **80/1,855 = 4.31%**; eligible/all known late
coverage is **80/137 = 58.39%**. These pooled daily results describe this historical
policy only. June informed model selection and is not untouched policy validation.
Zero denominators and unknown labels remain explicit in daily records.

## UI stability and month-end review — 24 October 2026 (C3/R1)

- Reproduced the reported hang locally: a slow `/predict/batch` call plus the
  shared `st.cache_data` lock left refreshed pages and other viewers waiting.
  Fixed with per-day score caching outside `st.cache_data`, short timeouts,
  chunked scoring with progress, and a clamped date range.
- Replay extended to June–August (18,960 orders). June orders and outcomes are
  byte-identical to the previous export.
- Month-end review reproduces the documented daily June result (80/137 late
  caught at 30%) and the report's monthly top-30% result (76/137).
- R1 accuracy against the Phase 2 target: June RMSE 7.369 / MAE 5.81 (matches
  the report), July 7.29 / 5.24, August 6.48 / 4.46.
- Daily list at 30%, share of all known late orders caught:

| Month | Late rate | C3 ranking | R1-buffer ranking | C3 ROC-AUC |
|---|---:|---:|---:|---:|
| June (selection) | 2.2% | 58.4% | 48.2% | 0.637 |
| July (monitoring) | 5.2% | 39.9% | 43.1% | 0.544 |
| August (monitoring) | 7.9% | 22.0% | 53.2% | 0.426 |

- 71 Phase 3 tests pass. All pages checked in the browser at 1440×900 against
  the live Render API.

## Initial API 2.0 deployment verified — 20 October 2026 (C3/R1)

- API: https://team8-delivery-api.onrender.com
- Swagger: https://team8-delivery-api.onrender.com/docs
- Service `team8-delivery-api`, free 0.1 CPU / 512 MB plan, Singapore region.
- Initial deployed source: `cd430e715fa798358436f1f5d1f725ead5256fba` on
  `classification-deployment`; initial Render deployment succeeded in 1m22s.
- Verified over public HTTPS: both models loaded, health, metadata, samples,
  Swagger, integrated single/batch prediction, both simulation date boundaries,
  retrospective endpoint, no future delivery timestamps in worklist, and 422
  responses for invalid date/input. The verification sequence took 39.704 seconds
  including requests during initial startup. This is not per-prediction latency.
- Independent comparison of hosted vs local fitted outputs is recorded in
  [render_model_equivalence.json](render_model_equivalence.json). The final synthetic-input
  endpoint verification is saved in [render_http_verification.json](render_http_verification.json)
  and took 73.232 seconds for the full sequence. Small floating
  differences from platform/BLAS arithmetic are checked with numerical tolerance.
- The default 2 June worklist returns 147 total/eligible, zero excluded and 45
  selected orders. R1 output is signed capped days from promise; C3 is a risk score.
- Python on this Mac initially lacked a usable default CA store. Verification used
  the installed certifi trusted CA bundle, with TLS certificate checking enabled.
- Initial hosted application startup and prediction succeeded. Idle-to-awake
  cold-start timing and sustained resource/load limits have not been measured.
- **Streamlit public deployment is pending at Andrew’s explicit request.** Its
  code and local verification remain complete; no public frontend URL is claimed.
- The separate course technical HTML guide was not found; the verified course
  PDF and professor examples were reviewed.

## Screenshots

Integrated UI screenshots are under [screenshots/](screenshots/).
The superseded classification-only guide and five original improved screenshots
are deliberately archived under [archive/classification-only/](archive/classification-only/).
