# Verification record

Verified locally on 19 October 2026 (Singapore), Python 3.13.5.

## Completed

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
  Reusable check: `python phase3/scripts/verify_http.py ACTUAL_BASE_URL`.
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

## Fixed daily-policy retrospective results

`python phase3/scripts/evaluate_daily.py` reproduces
[daily_policy_evaluation.json](daily_policy_evaluation.json). Defaults are 30%
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

## Live deployment not yet verified

- Render dashboard currently opens its sign-in page. No authenticated hosting
  access/API credential is available in this chat. `render.yaml`, Dockerfiles and
  startup configuration are ready; deploy the API first after account access.
- Streamlit Community Cloud also opens its sign-in page. After Render verification,
  configure `DELIVERY_API_URL` and deploy `phase3/frontend/app.py`.
- No public API/frontend URLs have been invented or claimed verified. Real HTTPS
  checks, hosted startup/cold-start, practical service resource limits and final
  connected frontend checks remain pending account access.
- The separate course technical HTML guide was not found; the verified course
  PDF and professor examples were reviewed. This does not block local implementation.

## Screenshots

Current integrated UI screenshots are under [screenshots/](screenshots/).
The superseded classification-only guide and five original improved screenshots
are deliberately archived under [archive/classification-only/](archive/classification-only/).
