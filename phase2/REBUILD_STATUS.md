# Simple → complex rebuild: status and handoff

**For:** whoever continues this work if the current session stops.
**Last updated:** 2026-10-09 (stage 2 in progress).

## Goal
Implement `phase2/PLAN_SIMPLE_TO_COMPLEX.md` (the repo copy is authoritative) on branch `simple-to-complex`. When every acceptance check in plan §9 passes, merge into `main` and push. The old notebooks must stay in `Archive/` so they are easy to restore. Phase 2 should end up holding only what the new report uses.

## Decisions made with the user
- Use the repo plan, not the Downloads copy. That means a BIC-stepwise supporting row (C2B/R2B), Lasso only in the overlap table, and a wide two-stage RF search.
- D6: plain logistic is unweighted. RF tunes `class_weight`. Thresholded metrics are reported at the top-10% cut-off and at 0.5. Add one CV sensitivity row for balanced logistic.
- Class-imbalance experiment in the classification notebook: random undersampling, random oversampling and SMOTENC, applied inside CV training folds only. It is reported, not adopted. If it clearly helps, ask the user before adopting it.
- Quasi-separation: in Train, `route_type == '3. Mixed'` (118 orders) and `seller_state_count >= 2` (227 orders) have zero late orders. Linear models use `merge_mixed_route=True` (in `src/linear_transforms.py`). `seller_state_count` is kept and flagged as not identified in the logistic tables. RF uses the raw features.
- Readable code (the repo may be graded). Every report diagram and evaluation table is generated (deciles → lift, model comparison, drift).
- After merging, archive every notebook and file that isn't used by the new report.
- Use Sonnet subagents for the implementation; the main session validates.

## Shared brief for the implementation agents
The full brief is in `REBUILD_AGENT_BRIEF.md`. Its key rules are summarised in the decisions above. Each notebook agent owns specific `src/` files:
- 01 owns `report_figures.py`, `validation_timelines.py`, `feature_selection.py` and the notes.
- 02 owns `evaluation.py`, `model_figures.py` and `resampling.py`.

Notebooks are executed with:
`../.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=-1`

## Progress
| Step | Status | Commit |
|---|---|---|
| Commit the multicollinearity screen and notes | done | 504e4a0 |
| Archive notebooks and `src/` to `Archive/2026-10-09_before_simple_to_complex/` | done | 05af09e |
| Stage 1 `src/` library (datasets, linear_transforms, variable_selection, rf_search, regression_models, interpretation, inference, slimmed tuning, evaluation) plus unittest suite (57 tests) | done | a84a80f |
| Install statsmodels, nbconvert, pytest and imbalanced-learn in `.venv`; update `requirements.txt` | done | a84a80f |
| RF search cache → `results/simple_to_complex/{task}/rf_best_params.json` | classification done (CV AP 0.2691 ± 0.0331); regression running | – |
| `01_data_features_multicollinearity.ipynb` (executed; log1p all 8 candidates; linear design 54 cols, max VIF 6.4 for log freight, kept pending CV in 02/03) | done | see git log |
| `02_classification.ipynb` (including the resampling experiment) | agent working | – |
| `03_regression.ipynb` | not started; begins after 02, to reuse `src/model_figures.py` | – |
| `04_summary`, `RESULTS_SUMMARY.md` and merged `run_metadata.json` | not started | – |
| Archive the remaining old notebooks and results; README rewrite. Remove from `phase2/` (copies are already in `Archive/2026-10-09_before_simple_to_complex/`): `classification_evaluation.ipynb`, `model_*_dev.ipynb`, `model_*_feature_selection.ipynb`, `model_validation_timelines.ipynb`, old `results/` runs | not started | – |
| Acceptance checks (plan §9): unittest, nbconvert execution of 01–04, assertions | not started | – |
| Merge into main and push | not started | – |

## How to resume
1. `git switch simple-to-complex` and read this file and the plan.
2. Check `git status`. Uncommitted notebooks or `src/` files may be partial agent work, so run them before committing.
3. Continue from the first step in the table that isn't done.
