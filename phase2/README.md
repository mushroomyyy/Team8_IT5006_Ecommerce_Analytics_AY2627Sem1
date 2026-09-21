# Phase 2 classification work

## Andrew's upload-ready notebook

Open `classification_problem_trial.ipynb` (Classification Problem — Trial Models & Findings): an executed companion adapted from Javier's
`model_dev.ipynb` flow, with attribution, methodology, findings, tables and five
embedded publication-style figures. Javier's notebook is unchanged. All 11 code
cells executed successfully; five PNG outputs and 15 PDF/SVG/PNG exports verified.

The notebook carries a fixed aggregate-results/curve snapshot (no individual
order IDs), so upload the `.ipynb` alone to share the rendered findings. Default
Run All regenerates figures without the local ignored prediction CSVs. It needs
NumPy, pandas, Matplotlib, Jinja2 and a Python notebook kernel. Full model training
is deliberately opt-in and requires the repository/data. Results are development
validation, not independent final testing. Rebuild the notebook after new runs;
its embedded snapshot does not silently refresh from disk.

Figures are in `andrew_findings_figures/`: serif/STIX mathematical typography,
vector PDF/SVG and 300-dpi PNG. Use PDF for LaTeX and PNG for slides/Word. No TeX
installation is needed to regenerate them.

To rebuild from current saved experiment outputs and execute:

```bash
.venv/bin/python -m pip install -r phase2/requirements-notebook.txt
.venv/bin/python phase2/build_findings_notebook.py --execute
```

The builder records source SHA-256 hashes, validates notebook/code syntax and
checks that the source notebook is unchanged. Review narrative conclusions when
updating experiment data: report prose is intentionally authored, not auto-inferred.

Start with [the implementation plan](CLASSIFICATION_IMPLEMENTATION_PLAN.md) and
[the current report draft](CLASSIFICATION_REPORT_DRAFT.md). The second modelling
problem remains undecided. Javier's `model_dev.ipynb` is the original reference.

Latest: [purchase/distance findings](PURCHASE_DISTANCE_FINDINGS.md) and
[ensemble results plus validation audit](ENSEMBLE_AND_VALIDITY_FINDINGS.md).
Run `.venv/bin/python phase2/classification_ensembles.py` after the purchase-distance
experiment for fixed soft voting and its paired AP comparisons. No final test has
been performed; dataset availability and population limitations are explicit.

For metric definitions and classification versus regression distinctions, read
[the evaluation guide](EVALUATION_METRIC_GUIDE.md). The
[expanded metric comparison](CLASSIFICATION_METRIC_COMPARISON.md) separates
ranking, forward decisions, probability quality and daily capacity.
Recreate it without retraining using
`.venv/bin/python phase2/classification_metric_report.py` after the ensemble run.

## Reproduce the corrected pilot

Purchase-time and geographic-distance follow-up:

```bash
.venv/bin/python phase2/classification_time_distance.py
```

Run this after `classification_experiments.py`. It preserves prior outputs and
compares each enriched model with purchase-time features, then purchase-time plus
distance features. Results and ignored local predictions use the separate
`classification_time_distance_*` filenames. See `PURCHASE_DISTANCE_FINDINGS.md`.

The follow-up feature/CatBoost ablation and paired inference are documented in
`CLASSIFICATION_EXPERIMENT_FINDINGS.md`. After generating the corrected pilot
predictions below, run:

```bash
.venv/bin/python phase2/classification_experiments.py
.venv/bin/python phase2/classification_significance.py
```

These commands use only the six development months, not May-June. The first
reuses the original logistic/forest/XGBoost predictions and checks cohort/label
alignment, adds base CatBoost, then refits all four models with 19 extra features.
It writes `classification_experiment_results.json` and local ignored predictions.
The second writes `classification_significance_results.json`: approximate paired
block-bootstrap AP comparisons with Holm correction and block-length sensitivity.
Cached baseline predictions must come from the current corrected backtest; rerun
that command if changing its inputs, feature construction or model settings.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r phase2/requirements.txt
.venv/bin/python -m unittest discover -s phase2 -p 'test_*.py'
.venv/bin/python phase2/classification_backtest.py --include-exploratory
```

On macOS, XGBoost may require `brew install libomp`. The command is run from the
repository root. Exact versions for the completed run are recorded in the results.

Without `--include-exploratory`, only the six November-April development folds run.
With it, May and June are additionally evaluated with monthly refitting and a
threshold fixed at May 1. These dates were already inspected and are not final
holdout data. No final holdout is evaluated by either command.

Outputs:

- `classification_backtest_results.json`: metrics, fold dates, prevalence,
  threshold source cutoffs, library versions and methodological limitations.
- `classification_backtest_predictions.csv`: paired order predictions, labels,
  label-maturity dates and prospective decisions; locally ignored by Git.

Leakage checks cover outcome-independent features, moving-window separation,
label maturity, and threshold selection without future outcomes. The historical
population is still restricted to eventual delivered orders. Generalisation to
all live orders, including cancellations, is not established. Related-table
snapshots are assumed available at approval because the data lack revision history.

The code evaluates daily capacity with expected random tie-breaking. It refits
monthly, not daily. It computes AP (not trapezoidal PR-AUC), ROC-AUC, F1, Brier,
log loss, coverage and daily-capacity lift. Calibration, hyperparameter search,
statistical inference, serialised deployment bundles and monitoring remain planned.

Earlier drafts and `pilot_model_results.json` are historical. Their single-split
tables must not be used as current evidence. `model_workbench.py` now contains
corrected shared helpers, so its legacy mixed classification/regression entry
point is not an exact reproduction of the old snapshot.
