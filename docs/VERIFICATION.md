# Verification record

Evidence that the notebooks, packaged models and live services for the selected
C2B and R2B models work as documented. The full test suite (`python -m pytest tests -q`)
passes 182 tests and 68 subtests.

## Full notebook rerun — 27 October 2026

All four notebooks ran sequentially, top to bottom in fresh kernels, in 375.2
seconds with cached Random Forest and decision-tree search results. No code cell
failed.

- C2B and R2B remain selected. Their saved per-split and CV metrics reproduce
  exactly; selected feature sets, operating threshold and fitted-state fingerprints
  are unchanged. Deployment artifact files are byte-identical and need no rebuild.
- Default classifier C3d's July AP differs by 7.31e-8 and ROC-AUC by 3.19e-7.
  Other model-metric differences are at floating-point precision (regression at
  most 3.56e-15). Reported rounded headline results are unchanged.
- July and August are evaluated on the same eligible orders that are scored:
  6,078 in July and 6,561 in August for both tasks.
- [Machine-readable rerun verification](notebook_rerun_verification.json).

## Packaged models — 26-27 October 2026

- The deployed models are C2B (`phase2-c2b-logistic-2018-06-02`, 13 BIC-selected
  features, operating cut-off 0.1282 from Validation) and R2B
  (`phase2-r2b-ols-2018-06-02`, 17 BIC-selected features), packaged as
  `deployment/artifacts/classification_model.joblib` and `regression_model.joblib`.
  They were chosen on Validation only; June is a retrospective out-of-sample
  assessment and July/August are monitoring. Canceled and unavailable orders are
  excluded from training and evaluation.
- The exporters refit both models on the June history and check them against the
  notebooks: June AP 0.08147 and June RMSE 7.3738 match the reference values
  exactly, and both fitted-state fingerprints equal the notebook fingerprints.
- A linux/amd64 Docker build of `deployment/Dockerfile.api` served both model
  versions on `/health`, and
  `python -m deployment.scripts.verify_http http://localhost:18002` passed all checks.
- C2B on June: average precision 0.0815, ROC-AUC 0.729; at the cut-off it flags
  6.3% of orders (34 of 116 known late caught); at 0.5 it flags none. R2B on June:
  RMSE 7.374, MAE 5.81 days. Monitoring ROC-AUC: July 0.591, August 0.560.
- [`daily_policy_evaluation.json`](daily_policy_evaluation.json) evaluates the
  deployed daily policy on June (30% follow-up, at least one day remaining,
  rounding up): 6,164 approvals, 6,134 known, 8 unknown, 116 known late;
  1,863 selected, 1,855 known, 3 unknown, 78 known late. Selected late rate
  78/1,855 = 4.20%; known late coverage 78/116 = 67.24%. The remaining 22
  approvals (5 of them selected) are canceled and have no outcome. The policy
  settings are demonstration assumptions.

## Live services — 27 October 2026

- `/health` on https://team8-delivery-api.onrender.com serves
  `phase2-c2b-logistic-2018-06-02` and `phase2-r2b-ols-2018-06-02`. `/model-info`
  exposes C2B's Validation-selected threshold, `0.12819778362486173`, and its policy.
- All eight live HTTP check groups pass: health, model metadata, single/batch
  agreement, independent/integrated agreement, invalid-input 422s, 404s for
  routes the API does not serve, inference-only OpenAPI/metadata and Swagger.
  Hosted predictions for the synthetic example match the local packaged models.
  TLS certificate checking stayed enabled.
- Delivery Watch (https://team8-delivery-watch.streamlit.app/) shows
  **API connected · C2B + R2B**. Models and guidance renders the 13-feature C2B
  and 17-feature R2B identities, model-selection tables, confusion metrics and
  false-negative analysis.
- Daily simulation scores all 129 eligible approvals for 2 July 2018 using the
  live API, with 39 selected at the default 30% follow-up proportion; outcomes
  stay hidden until revealed.
- [Machine-readable live verification](render_c2b_r2b_verification.json).
