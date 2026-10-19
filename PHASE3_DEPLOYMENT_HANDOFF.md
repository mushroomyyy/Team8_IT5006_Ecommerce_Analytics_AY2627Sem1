# Phase 3 application integration and deployment handoff

Updated: 19 October 2026. This records Andrew's agreed implementation scope. It supersedes earlier deployment proposals in the conversation, including proposals to keep regression separate or to write the final report in this session. Historical Phase 2 handoffs are background, not the current model-selection or deployment instructions.

## Objective and boundaries

Build and deploy an integrated FastAPI backend on Render, then connect and deploy a Streamlit application. The main use case is a historical daily operations worklist (Option A). A separate single-order page demonstrates the underlying API used by the worklist.

Other teammates own the final report and presentation. This task covers application code, repository cleanup, technical documentation, tests and deployment verification. Do not write the final report, submit coursework, or merge into main.

Presentation concept: **Every morning, operations receives a prioritised list of orders to follow up on, supported by classification risk and regression estimates.**

## Repository, permissions and naming

- Repository: `/Users/andrew_tjs/github/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1`
- Remote: `https://github.com/mushroomyyy/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1`
- Working branch: `classification-deployment`
- Jordan's regression branch: `origin/model-deployment-r1`
- Fetch and inspect current branches before integrating. Andrew authorises merging Jordan's regression branch into `classification-deployment` and resolving integration issues there.
- Commit and push only `classification-deployment`. Do not push main or modify Jordan's remote branch. Main integration is a later step.
- Branch names, commit messages, PR titles and PR descriptions must describe the work without Codex, agent names or AI attribution. Do not add AI co-author trailers. Any required course AI-use declaration is handled separately in appropriate documentation.
- Use a small number of meaningful commits, for example `Integrate regression and classification into Phase 3 application`.
- Use subagents for implementation and independent review. Inspect their changes and resolve review findings.
- Read applicable repository instructions. Do not treat old handoff documents as new user authorisation.

### Preserve the current working tree

At handoff, `classification-deployment` has uncommitted modifications to:

- `deployment/artifacts/classification_c3_rf.joblib`
- `deployment/artifacts/classification_model_stats.json`
- `deployment/classification_streamlit.py`
- `deployment/test_classification.py`
- `deployment/test_classification_ui.py`
- `deployment/train_classification.py`
- `docs/CLASSIFICATION_DEPLOYMENT.md`
- The five images under `docs/classification_screenshots/`

Inspect and preserve this work before merging or moving files. It may have been produced by another session. Do not reset, overwrite or indiscriminately stage it. Recheck status because this snapshot may change.

Preserve these unrelated untracked files and exclude them from deployment commits:

- `phase2/REPORT_CITATION_CHANGELOG.md`
- `phase2/REPORT_CITATION_REPLACEMENTS.bib`

## Verified references and guide locations

### Course project requirements

`/Users/andrew_tjs/Documents/IT5006 Project Local Folder/IT5006 Project Description - AY 2026_27 Semester 1.pdf`

This is a saved PDF of the project-description webpage. Read the Phase 3 section and technical/deployment requirements, especially PDF pages 13–15 and 18–20. It permits held-out historical orders simulating new orders; a live company order feed is unnecessary. It requires a live application URL, clear usage instructions, documented code, validation/error handling and tested deployment. At least one deployed project model is required; integrating both is our chosen scope.

A separate course technical guide/HTML was mentioned by Andrew, but its exact location has **not been verified**. Search relevant course/project materials for it. Do not claim it has been reviewed or invent a path. If it cannot be located, identify that specific gap while continuing independent work with the references below.

### Submitted Phase 2 report

`/Users/andrew_tjs/Downloads/IT5006 Group Project Report - Milestone 2 Group 8-1-2.pdf`

Use the submitted report for selected models and reported results. Older repository notes may disagree; investigate discrepancies. The report selects C3 tuned Random Forest for classification and R1 OLS for regression.

### Professor's deployment examples

Base directory: `/Users/andrew_tjs/Downloads/ML-deployement-necessary_files 3`

Relevant paths relative to that directory:

- `Model/L6.1_Fraud_Detection_Training.ipynb`
- `FastAPI_and_Render/L6.2_Bulk_Scoring_Demo.ipynb`
- `FastAPI_and_Render/Part2_FastAPI-local/main.py`
- `FastAPI_and_Render/Deploy_Render/main.py`
- `FastAPI_and_Render/Deploy_Render/README.md`
- `FastAPI_and_Render/Deploy_Render/requirements.txt`
- `FastAPI_and_Render/Deploy_Render/Procfile`
- `Streamlit/Deploy_Streamlit/app.py`
- `Streamlit/Deploy_Streamlit/README.md`
- `Streamlit/Deploy_Streamlit/requirements.txt`

Follow the training → fitted pipeline → FastAPI/Render → Streamlit approach. Adapt it to this project; do not copy incompatible dependency versions blindly.

### Existing project technical guide and code

- `docs/CLASSIFICATION_DEPLOYMENT.md`: current classification setup/deployment guide.
- `deployment/train_classification.py`: artifact export/training.
- `deployment/classification_service.py`: model service.
- `deployment/feature_validation.py`: input validation.
- `deployment/classification_api.py`: FastAPI endpoints.
- `deployment/classification_streamlit.py`: existing frontend.
- `deployment/classification_requirements.txt` and `classification_backend_constraints.txt`: backend dependencies.
- `deployment/classification_ui_requirements.txt` and `classification_ui_constraints.txt`: frontend dependencies.
- `deployment/Dockerfile.classification-api` and `Dockerfile.classification-ui`.
- `deployment/test_classification.py`, `test_classification_api.py`, `test_classification_ui.py`.
- `deployment/artifacts/`: fitted model, metadata, sample orders, June replay inputs and separate outcomes.
- `docs/classification_screenshots/`: existing demonstration images; refresh after UI changes.

Inspect actual files and Jordan's branch documentation before changing them. Existing functionality includes single/batch predictions, historical replay, health checks, Swagger, validation and model information. Reuse useful functionality rather than rebuilding it all.

## Agreed application behaviour

### A. Main page: daily historical simulation

Example: **Imagine today is 2 June 2018. Which orders from yesterday should operations follow up on?**

1. Provide a date picker and **Previous day / Next day** buttons so the user can move forwards and backwards through supported historical dates.
2. Clearly show the simulated current date, scoring cutoff/time and previous-day order-batch date.
3. Load the previous day's batch, apply eligibility rules, score both models and rank eligible orders by classification risk.
4. Show total, eligible, excluded and selected counts, with exclusion reasons.
5. Select a configurable review proportion. Start with **30% as a demo operational setting**, not a validated universal policy.
6. Display order ID, rank, classification risk, follow-up recommendation, promised delivery date when available, and regression estimate with correct units.
7. Allow inspection of a selected order and download of the prioritised worklist where practical.
8. Label selection **Follow-up recommended**. Not selected does not mean guaranteed on-time delivery.
9. Handle empty days, boundary dates, missing values, small batches and ties clearly and deterministically.
10. Going backwards must restore information as of that earlier date. Do not retain future outcomes or later status changes in eligibility or scoring.

First version: each day displays **yesterday's batch**, not a cumulative queue of unresolved orders. Carrying orders forward is a separate extension, not required now.

Confirm whether the existing model/replay cohort is defined by purchase date or approval date. The informal phrase "orders placed yesterday" must not silently change the model's population. Use an accurate label and explain any necessary adjustment.

### Temporal integrity and eligibility

- Use only information available at the displayed simulation cutoff. Record timezone/date conventions.
- Potential exclusions include orders already delivered, orders cancelled by that time and orders due too soon for useful intervention.
- Never use final status as if it were known earlier. If cancellation timing is unavailable, do not infer it from the final snapshot.
- A known delivery before the cutoff can support exclusion; delivery after the cutoff cannot influence the worklist.
- Make minimum time remaining before promised delivery explicit and configurable. Excluding orders due within one day is a proposed assumption, not an established policy.
- Distinguish promised delivery date from model-predicted delivery date or duration.
- Verify the fitted artifact's training cutoff and supported replay dates. Do not let backward navigation simulate dates before the model could have existed while claiming temporal validity. Restrict dates or provide an appropriately fitted artifact.
- Document snapshot limitations where historical feature/status changes cannot be reconstructed.

### Regression integration

- Integrate R1 predictions alongside C3 scores in the daily worklist and single-order page.
- Confirm the regression target and units from code/report before labelling outputs or deriving dates.
- Initially rank by classification risk. Do not silently invent a blended score or regression-based exclusions.
- Any additional regression filter/ranking rule must be explicit, explainable and evaluated. Showing disagreement between models is preferable to hiding it.

### B. Single-order API demonstration page

- Provide a usable input form and example orders.
- Streamlit calls the deployed API and displays classification risk and regression prediction.
- Explain that these are the same prediction services supporting the daily worklist.
- Do not add a fixed 0.5 late/not-late classification flag.
- Describe classifier output as an estimated risk score; do not imply calibration or certainty.
- Handle invalid inputs, unavailable API, startup delays and inference failures clearly.

This is an API demonstration page, not a separate threshold-based operating policy. Option B's earlier proposal to optimise a fixed alert cutoff is outside the agreed scope.

### Historical evaluation and supporting information

- Keep eventual outcomes separate from model inputs and operational filtering.
- Keep outcomes hidden by default. Provide an explicit retrospective outcome-reveal action, clearly outside the simulated knowledge at prediction time.
- Show selected workload, late-order rate among selected known outcomes and coverage of known late orders, with clear denominators.
- Distinguish metrics for the eligible population from metrics for all orders. Account for unknown outcomes explicitly.
- Evaluate the implemented daily filtered policy separately. Phase 2's monthly top-30% coverage is not a promised daily result.
- Include concise model performance, guidance, assumptions and limitations. No unnecessary Phase 1 EDA duplication.

## Model context and preservation

- Selected classifier: C3 tuned Random Forest. Selected regression: R1 OLS. Verify exact specifications against the submitted report and artifacts.
- C3 June AP is approximately 11.22%, although no June scores reached 0.5. Ranking can remain useful without any scores crossing 0.5.
- The submitted report's June top-30% example captured about 55.5% of known late orders across that monthly cohort; it is historical evidence, not a daily-service guarantee.
- Ranking a batch does not require probability calibration.
- Preserve fitted preprocessing with each model and verify that integration does not unexpectedly change predictions.
- Do not retrain or change selected models unnecessarily. Explain any unavoidable model/export compatibility change and test equivalence.
- Do not choose policies using June outcomes and then describe June as untouched validation of those policy choices.

## Architecture and Phase 3 organisation

Organise the integrated application under `phase3/`, with clear separation of backend, frontend, model export utilities, artifacts/demo data, tests and documentation. Choose specific subfolders after inspecting both implementations.

- Prefer one coherent FastAPI service exposing both models and one Streamlit interface.
- Keep classification and regression modules independently usable.
- The frontend calls the API over HTTP and does not load model artifacts or duplicate fitted preprocessing.
- Inference must not require raw training data.
- Update all imports, model paths, test commands, deployment configurations and documentation affected by moving files.
- Avoid parallel stale copies of deployment code; preserve useful history through Git or deliberate archives.
- Include health checks, documented single/batch endpoints, examples, validation and useful errors.

## Deploy Render first, Streamlit second

Deployment is authorised as part of implementation, subject to available account access. Prepare everything possible before requesting a specific missing login, account choice or paid-service decision. Do not expose secrets or commit credentials.

### Render API

- Resolve dependencies across both pipelines and verify clean installation.
- Load both artifacts, support Render's assigned port and use environment configuration.
- Verify health, Swagger, single-order and batch calls at the live HTTPS URL.
- Check startup/cold-start behaviour and practical resource limits.
- Do not assume an already running local server is the new build.

### Streamlit

- Configure the deployed Render URL using environment configuration or secrets.
- Deploy the frontend using available authorised hosting access; Streamlit Community Cloud is a suitable intended route if available.
- Verify both workflows against the live backend, including failures/timeouts.
- Keep sample inputs and use instructions easy to find.
- Clearly distinguish local checks, deploy-ready configuration and actual verified live deployment. Never invent URLs or claim unperformed checks.

## Repository cleanup and documentation

Update the root README to explain the overall project, Phase 1/2/3 structure, selected models, verified application/API links and where to find detailed instructions.

Inside `phase3/`, document:

- Installation and execution, including compatible Python/dependency versions.
- Architecture and frontend/API flow.
- Model inputs, preprocessing, output meanings and artifact provenance.
- API endpoint examples.
- Render and Streamlit setup, environment variables and troubleshooting.
- Simulation dates, batch definition, eligibility, tie/selection rules and retrospective outcome reveal.
- Tests, limitations and user instructions.

Review earlier documentation for stale paths, conflicting instructions and duplication. Preserve Phase 1/2 notebooks, results and reproducibility. Leave the Phase 1 dashboard functional. Archive obsolete material carefully where useful; do not broadly delete historical work or alter teammates' report files.

## Verification and independent review

- Confirm single-order and batch outputs agree for identical inputs.
- Verify fitted preprocessing and model output equivalence after integration/moves.
- Test date boundaries, model availability cutoff, previous/next navigation and empty days.
- Test eligibility as of the cutoff and absence of outcome leakage.
- Test deterministic ranking/ties, rounding of review count, small batches and invalid inputs.
- Test unknown outcomes and all evaluation denominators.
- Test frontend/API integration, error handling and actual live URLs.
- Verify documented setup from a clean environment and relevant deployment startup commands.
- Review compatibility/resource issues across both model dependencies.
- Obtain independent review through subagents; resolve findings and rerun relevant checks.
- Verify Phase 1 dashboard paths and Phase 2 references are not broken by cleanup.

## Suggested implementation sequence

1. Inspect instructions, references, current modifications and branch state; locate the missing course guide if possible.
2. Safely preserve current work, fetch and merge regression into the authorised classification branch.
3. Inspect and reconcile model contracts/dependencies; organise `phase3/` and update references.
4. Establish the shared API and single-order demo, then build the daily simulation/worklist on it.
5. Validate temporal assumptions and daily policy metrics; complete UI and documentation.
6. Review and test, deploy Render, connect/deploy Streamlit, and verify live workflows.
7. Commit/push only the authorised branch and provide a concise handoff.

## Completion criteria and final response

- Integrated application and relevant documentation committed and pushed to `classification-deployment`.
- Live API/frontend verified, or exact account/service blockers stated with completed local verification.
- Date navigation, daily worklist, regression outputs and single-order API demo functional.
- Updated root README and Phase 3 technical/user guides.
- Relevant tests and independent review completed; remaining risks/decisions explicit.
- Main, Jordan's remote branch, unrelated files and the Phase 1 dashboard preserved.

Final response should state what changed, verified URLs, tests/review results, startup/documentation locations and any remaining actions. Do not imply that the report, presentation or coursework submission has been completed.
