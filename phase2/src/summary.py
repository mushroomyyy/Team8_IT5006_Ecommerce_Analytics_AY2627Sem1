"""Cross-task summary of the simple-to-complex pipeline, built only from exported files.

Reads the tables, `model_results_long.csv`, `cv_fold_scores.csv` and `run_metadata.json` that
notebooks 02 and 03 wrote under `results/simple_to_complex/{classification,regression}/`. No
model is refitted. It writes:

* `figures/04_*.png`: the simple-to-complex ladder, the Train - CV gaps and a drift summary;
* `RESULTS_SUMMARY.md`: every table the report needs, generated from the exports;
* `run_metadata.json`: both tasks' metadata plus package versions.

Run it with `python -m src.summary` or from `04_summary.ipynb`.
"""
import argparse
import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import model_figures as mf
from . import report_figures as figs

DEFAULT_ROOT = Path('results/simple_to_complex')
LADDER_SPLITS = ('CV', 'Validation', 'June')
PERIODS = ('June', 'July', 'August')
SCORE_COLUMNS = {'mean_score', 'sd_score', 'mean_fit_time', 'mean_score_time'}
PREVIOUS_PIPELINE = {'model': 'LightGBM', 'june_ap_percent': 15.09, 'june_mae_days': 4.85}
SPLIT_LABELS = {'June': 'Test (June)', 'July': 'Monitoring (July)', 'August': 'Monitoring (August)'}


@dataclass(frozen=True)
class TaskSpec:
    """Everything that differs between the two tasks."""
    name: str
    title: str
    rungs: tuple                # (rung label, model key), simplest first
    metric: str                 # primary metric column in model_results_long.csv
    metric_label: str
    scale: float                # multiply the stored metric by this for display
    score_sign: float           # sign of the RF search score (regression stores negative RMSE)
    effect_stem: str            # exported coefficient table stem
    effect_is_ratio: bool       # odds ratios (True) or days (False)
    linear_models: tuple
    forest: str
    forest_default: str


TASKS = {
    'classification': TaskSpec(
        name='classification', title='Classification: will the order arrive late?',
        rungs=(('Baseline', 'C0'), ('Linear, all', 'C1'), ('Linear, selected', 'C2'),
               ('RF, default', 'C3d'), ('RF, tuned', 'C3')),
        metric='avg_precision', metric_label='Average precision (AP, %)', scale=100.0, score_sign=1.0,
        effect_stem='classification_odds_ratios', effect_is_ratio=True, linear_models=('C1', 'C2'), forest='C3', forest_default='C3d',
        ),
    'regression': TaskSpec(
        name='regression', title='Regression: days from the promised delivery date',
        rungs=(('Baseline', 'R0a'), ('Linear, all', 'R1'), ('Linear, selected', 'R2'),
               ('RF, default', 'R3d'), ('RF, tuned', 'R3')),
        metric='rmse', metric_label='RMSE (days)', scale=1.0, score_sign=-1.0,
        effect_stem='regression_coefficients', effect_is_ratio=False, linear_models=('R1', 'R2'), forest='R3', forest_default='R3d',
        ),
}


@dataclass
class TaskExports:
    """One task's exported files, loaded read-only."""
    spec: TaskSpec
    directory: Path
    metadata: dict
    results: pd.DataFrame
    folds: pd.DataFrame

    def table(self, stem, title_contains):
        """Read the exported table of `{stem}.html` whose title contains `title_contains`, as text."""
        titles = table_titles(self.directory / f'{stem}.html')
        for number, title in enumerate(titles, 1):
            if title_contains in title:
                return pd.read_csv(self.directory / f'{stem}_{number}.csv', dtype=str, keep_default_na=False)
        raise KeyError(f'No table titled like {title_contains!r} in {stem}.html: {titles}')

    def csv(self, name):
        return pd.read_csv(self.directory / name)


def table_titles(html_path):
    """Titles of the tables in an `export_report_tables` HTML file, in export order."""
    return [unescape(t) for t in re.findall(r'<p><b>(.*?)</b></p>', Path(html_path).read_text(encoding='utf-8'))]


def load_task(name, root=DEFAULT_ROOT):
    """Load a task's metadata, per-split results and per-fold CV scores."""
    spec = TASKS[name]
    directory = Path(root) / name
    return TaskExports(spec, directory, json.loads((directory / 'run_metadata.json').read_text()),
                       pd.read_csv(directory / 'model_results_long.csv'),
                       pd.read_csv(directory / 'cv_fold_scores.csv'))


# ---------------------------------------------------------------- ladder numbers

def ladder_frame(task):
    """Primary metric per rung and split (CV, Validation, June), with the CV SD, in display units."""
    spec, rows = task.spec, []
    cv = task.folds.groupby('model')[spec.metric].agg(['mean', 'std'])
    june_val = task.results.set_index(['model', 'split'])[spec.metric]
    for rung, model in spec.rungs:
        rows.append({'rung': rung, 'model': model, 'split': 'CV', 'value': cv.loc[model, 'mean'] * spec.scale,
                     'sd': cv.loc[model, 'std'] * spec.scale})
        rows += [{'rung': rung, 'model': model, 'split': split,
                  'value': june_val.loc[(model, split)] * spec.scale, 'sd': np.nan} for split in LADDER_SPLITS[1:]]
    return pd.DataFrame(rows)


def ladder_markdown_table(task):
    """Wide ladder table: one row per rung, one column per split."""
    frame = ladder_frame(task)
    wide = frame.pivot(index=['rung', 'model'], columns='split', values='value')
    sd = frame[frame['split'] == 'CV'].set_index(['rung', 'model'])['sd']
    rows = []
    for rung, model in task.spec.rungs:
        rows.append([rung, model, f"{wide.loc[(rung, model), 'CV']:.2f} +/- {sd.loc[(rung, model)]:.2f}",
                     f"{wide.loc[(rung, model), 'Validation']:.2f}", f"{wide.loc[(rung, model), 'June']:.2f}"])
    return md_table(['Rung', 'Model', 'CV mean +/- SD', 'Validation', 'Test (June)'], rows)


# ---------------------------------------------------------------- figures

def _rung_ticks(task):
    return [f'{rung}\n{model}' for rung, model in task.spec.rungs]


def plot_ladder(tasks, save_path=None):
    """Primary metric along the simple-to-complex ladder on CV, Validation and Test (June)."""
    styles = {'CV': (figs.BLUE, 'o'), 'Validation': (figs.AQUA, 's'), 'June': (figs.ORANGE, 'D')}
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 5))
        for ax, task in zip(np.atleast_1d(axes), tasks):
            frame = ladder_frame(task)
            for split, (color, marker) in styles.items():
                part = frame[frame['split'] == split]
                ax.errorbar(range(len(part)), part['value'], yerr=part['sd'].to_numpy() if split == 'CV' else None,
                            color=color, marker=marker, ms=6, capsize=3, lw=1.6,
                            label='Test (June)' if split == 'June' else split)
            ax.set_xticks(range(len(task.spec.rungs)), _rung_ticks(task), fontsize=8.5)
            ax.set(xlabel='Model rung, simple to complex', ylabel=task.spec.metric_label,
                   title=task.spec.title)
            ax.legend()
            figs._style(ax, 'y')
        return mf._finish(fig, 'Simple-to-complex ladder: primary metric by evaluation split', save_path,
                          'CV error bars are +/- 1 SD over 5 chronological folds. No model is selected.')


def plot_overfitting_gaps(tasks, save_path=None):
    """Train minus CV mean gap of the primary metric for every rung."""
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8))
        for ax, task in zip(np.atleast_1d(axes), tasks):
            spec = task.spec
            gaps = task.table(f'{spec.name}_overfitting_gaps', 'overfitting gap').set_index('model')
            values = [float(gaps.loc[model, spec.metric]) * spec.scale for _, model in spec.rungs]
            ax.bar(range(len(values)), values, color=[mf.MODEL_COLORS[m] for _, m in spec.rungs])
            ax.axhline(0, color=figs.INK2, lw=0.8)
            ax.set_xticks(range(len(values)), _rung_ticks(task), fontsize=8.5)
            unit = 'percentage points' if spec.scale == 100 else 'days'
            ax.set(xlabel='Model rung, simple to complex', ylabel=f'Train minus CV mean ({unit})', title=spec.title)
            figs._style(ax, 'y')
        return mf._finish(fig, 'Overfitting gaps along the ladder', save_path,
                          'AP: positive means Train is better than CV. RMSE: negative means Train error is lower.')


def plot_drift_summary(tasks, save_path=None):
    """Primary metric from Test (June) to Monitoring (August), and the change over those months."""
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(2, len(tasks), figsize=(6.2 * len(tasks), 8.4))
        for column, task in enumerate(tasks):
            spec = task.spec
            drift = task.table(f'{spec.name}_drift', 'drift')
            drift = drift[drift['metric'] == spec.metric].set_index('model')
            top, bottom = axes[0, column], axes[1, column]
            for _, model in spec.rungs:
                row = drift.loc[model]
                top.plot(list(PERIODS), [float(row[p]) * spec.scale for p in PERIODS], color=mf.MODEL_COLORS[model],
                         marker=mf.MODEL_MARKERS.get(model, 'o'), label=model)
            top.set(xlabel='Month (June = Test, July and August = Monitoring)', ylabel=spec.metric_label,
                    title=spec.title)
            top.legend(ncol=2)
            change = [float(drift.loc[m, 'change_june_to_last']) * spec.scale for _, m in spec.rungs]
            bottom.bar(range(len(change)), change, color=[mf.MODEL_COLORS[m] for _, m in spec.rungs])
            bottom.axhline(0, color=figs.INK2, lw=0.8)
            bottom.set_xticks(range(len(change)), _rung_ticks(task), fontsize=8.5)
            unit = 'percentage points' if spec.scale == 100 else 'days'
            bottom.set(xlabel='Model rung, simple to complex', ylabel=f'August minus June ({unit})')
            figs._style(top, 'y')
            figs._style(bottom, 'y')
        return mf._finish(fig, 'Drift of frozen models from June to August', save_path,
                          'Late rates differ by month, so the baseline also moves.')


# ---------------------------------------------------------------- markdown helpers

def md_table(headers, rows):
    """GitHub-style pipe table from a header list and rows of cell values."""
    def cell(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(cell, headers)) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]
    return '\n'.join(lines)


def frame_markdown(frame):
    return md_table(list(frame.columns), frame.itertuples(index=False, name=None))


def top_effects(table, spec, n=8):
    """Largest standardised effects (per 1 SD where defined, else per unit or vs reference level).

    Ranked by |log odds ratio| or |days|. Terms without an effect (sine/cosine and squared terms,
    unidentified terms) are skipped; callers list those separately.
    """
    prefix = 'OR' if spec.effect_is_ratio else 'Days'
    sd_col, unit_col = f'{prefix} per 1 SD', f'{prefix} per unit'
    rows = []
    for row in table.to_dict('records'):
        use_sd = row[sd_col] != ''
        value = row[sd_col] if use_sd else row[unit_col]
        if value == '':
            continue
        size = abs(np.log(float(value))) if spec.effect_is_ratio else abs(float(value))
        rows.append((size, [row['Term'], 'per 1 SD' if use_sd else row['Basis'], value,
                            row['95% CI per 1 SD' if use_sd else '95% CI per unit'], row['p-value']]))
    rows.sort(key=lambda item: -item[0])
    return [r for _, r in rows[:n]]


def rf_sensitivity(stage_a, spec):
    """Per hyperparameter: best and worst value by the best Stage A candidate that used it."""
    rows = []
    for parameter in [c for c in stage_a.columns if c not in SCORE_COLUMNS and not c.startswith('split')]:
        best = stage_a.assign(value=stage_a[parameter].fillna('None').astype(str)).groupby('value')['mean_score'].max()
        shown = best * spec.score_sign
        top, bottom = best.idxmax(), best.idxmin()
        rows.append((float(best.max() - best.min()), [parameter, top, f'{shown[top]:.4f}', bottom,
                                                      f'{shown[bottom]:.4f}', f'{abs(best.max() - best.min()):.4f}']))
    rows.sort(key=lambda item: -item[0])
    return md_table(['Hyperparameter', 'Best value', 'Best candidate score', 'Worst value', 'Worst candidate score',
                     'Spread'], [r for _, r in rows])


# ---------------------------------------------------------------- task sections

class Numbered:
    """Hands out '2.1', '2.2', ... subsection numbers for one task."""

    def __init__(self, chapter):
        self.chapter, self.count = chapter, 0

    def __call__(self, title):
        self.count += 1
        return f'### {self.chapter}.{self.count} {title}'


def _row_counts(task):
    meta = task.metadata['row_counts']
    scored = md_table(['Evaluation split', 'Orders scored'],
                      [[SPLIT_LABELS['June'], f"{meta['june_scored']:,}"], [SPLIT_LABELS['July'], f"{meta['july_scored']:,}"],
                       [SPLIT_LABELS['August'], f"{meta['august_scored']:,}"]])
    counts = frame_markdown(task.csv('row_counts.csv').astype(str))
    return [counts, '', 'Frozen models score the Test and Monitoring months:', '', scored]


def _transforms(task):
    meta = task.metadata['linear_transforms']
    table = task.table(f'{task.spec.name}_transform_cv', 'Transformation')
    keep = [c for c in table.columns if c not in ('groups',)]
    names = {'step': 'Decision step', 'design': 'Design', 'kept': 'Kept'}
    table = table[keep].rename(columns=names)
    text = (f"Groups kept: {', '.join(meta['groups_kept'])}. Log columns: {', '.join(meta['log_cols'])}. "
            f"Squared columns: {', '.join(meta['squared_cols'])}. Mixed route folded into All interstate: "
            f"{meta['merge_mixed_route']}.")
    return [text, '', frame_markdown(table)]


def _stepwise(task):
    spec, stepwise = task.spec, task.metadata['stepwise']
    path = task.table(f'{spec.name}_stepwise_path', 'CV-stepwise path').copy()
    path['Marker'] = ''
    path.loc[path['step'] == str(stepwise['best_step']), 'Marker'] = 'best CV step'
    path.loc[path['step'] == str(stepwise['one_se_step']), 'Marker'] += ' 1-SE choice'
    selected = task.metadata['selected_variables']
    chosen = spec.linear_models[1]
    text = (f"The best CV step is {stepwise['best_step']} and the 1-SE rule chooses step {stepwise['one_se_step']}, "
            f"which gives {chosen} with {len(selected[chosen])} variable(s): {', '.join(selected[chosen])}.")
    return [text, '', frame_markdown(path.rename(columns={'cv_mean': 'CV mean', 'cv_se': 'CV SE'}))]


def _overlap(task):
    selected = task.metadata['selected_variables']
    counts = md_table(['Method or model', 'Variables selected'],
                      [[m, len(selected[m])] for m in selected if m != task.spec.linear_models[0]])
    table = task.table(f'{task.spec.name}_selection_overlap', 'overlap')
    table = table.replace({'True': 'yes', 'False': ''})
    return [counts, '', frame_markdown(table)]


def _rf_search(task):
    spec, meta = task.spec, task.metadata
    space = task.table(f'{spec.name}_rf_search_summary', 'search space')
    summary = task.table(f'{spec.name}_rf_search_summary', 'search summary')
    sensitivity = rf_sensitivity(task.csv('rf_search_stageA.csv'), spec)
    return [f"Stage A drew {meta['rf_search']['n_iter_stage_a']} random candidates from this space.", '',
            frame_markdown(space), '', frame_markdown(summary), '',
            'Sensitivity of the Stage A search (spread = best minus worst score over the values of one hyperparameter; '
            'a larger spread means the score is more sensitive to that hyperparameter):', '', sensitivity]


def _evaluation_tables(task):
    spec = task.spec
    html = task.directory / f'{spec.name}_model_comparison.html'
    out = []
    for title in table_titles(html):
        table = task.table(f'{spec.name}_model_comparison', title)
        out += [f'**{title}**', '', frame_markdown(table.rename(columns={'June': 'Test (June)'})), '']
    return out


def _effects(task):
    spec, out = task.spec, []
    for model in spec.linear_models:
        table = task.table(spec.effect_stem, f'{model} ')
        header = ['Term', 'Scale', 'OR' if spec.effect_is_ratio else 'Days', '95% CI', 'p-value']
        out += [f'**{model}: largest effects**', '', md_table(header, top_effects(table, spec)), '']
        if 'Identification' in table.columns:
            for row in table[table['Identification'] != ''].itertuples():
                out += [f'{row.Term}: {row.Identification}.', '']
    return out


def _rf_top_features(task):
    spec = task.spec
    agreement = task.table(f'{spec.name}_rf_agreement', 'Random Forest top features')
    june = task.csv('rf_permutation_importance_june.csv').head(5)
    june_rows = [[int(r.rank), r.feature, f'{r.importance_mean:.4f}', f'{r.importance_sd:.4f}'] for r in june.itertuples()]
    return ['**Top 10 by permutation importance on Validation, with linear-model significance**', '',
            frame_markdown(agreement), '', '**Top 5 by permutation importance on Test (June)**', '',
            md_table(['Rank', 'Feature', 'Importance (mean)', 'Importance (SD)'], june_rows)]


def _figure_links(root, names):
    out = []
    for name, caption in names:
        if not (Path(root) / 'figures' / name).exists():
            raise FileNotFoundError(f'Missing figure {name}; run the notebooks first.')
        out += [f'![{caption}](figures/{name})', '']
    return out


def _classification_extras(task, root, number):
    weights = task.table('classification_class_weight_sensitivity', 'class-weight')
    resampling = task.table('classification_resampling', 'Class-imbalance')
    shown = resampling[['Model', 'Option', 'CV AP', 'CV Brier', 'Delta CV AP vs none (paired, +/- SE)',
                        'Validation AP', 'Validation Brier']]
    meta = task.metadata['resampling']
    verdict = 'adopted' if meta['adopted'] else 'not adopted'
    figures = [('02_roc_curves.png', 'ROC curves, Validation and June'),
               ('02_pr_curves.png', 'Precision-recall curves, Validation and June'),
               ('02_calibration.png', 'Calibration'), ('02_june_decile_lift.png', 'June risk deciles: capture and lift')]
    return [number('Class-weight sensitivity'), '', frame_markdown(weights), '',
            number('Class-imbalance experiment'), '',
            f"Resampling happened inside the CV training folds only. Result: {verdict}; "
            f"a clear CV AP gain for any option: {meta['clear_ap_gain_any']}.", '', frame_markdown(shown), '',
            number('ROC, PR, calibration and decile figures'), '', *_figure_links(root, figures)]


def _regression_extras(task, root, number):
    by_outcome = task.table('regression_error_by_outcome', 'late versus on-time')
    late_flag = task.table('regression_late_flag', 'derived late flag')
    figures = [('03_predicted_vs_actual.png', 'Predicted versus actual days'),
               ('03_ols_diagnostics.png', 'R1 diagnostics')]
    return [number('Error by outcome and derived late flag'), '',
            '**RMSE and MAE for late versus on-time orders (days)**', '', frame_markdown(by_outcome), '',
            '**Derived late flag (prediction > 0 days) versus the classification label**', '', frame_markdown(late_flag), '',
            number('Figures'), '', *_figure_links(root, figures)]


def task_section(task, chapter, root):
    """All of one task's plan-9 content as markdown lines."""
    spec, number = task.spec, Numbered(chapter)
    prefix = f'{spec.name}_'
    blocks = [
        ('Row counts per split', _row_counts(task)),
        ('Transforms kept (linear models), with CV evidence', _transforms(task)),
        ('Stepwise path, chosen step and selected variables', _stepwise(task)),
        ('Selection overlap: CV-stepwise, BIC, AIC and Lasso', _overlap(task)),
        ('Random Forest search and sensitivity', _rf_search(task)),
        ('Evaluation tables (Train, CV, Validation, Test (June), Monitoring)', _evaluation_tables(task)),
        ('Success criteria', [frame_markdown(task.table(prefix + 'success_criteria', 'success criteria'))]),
        ('Overfitting gaps (Train minus CV mean)',
         [frame_markdown(task.table(prefix + 'overfitting_gaps', 'overfitting gap'))]),
        ('Random Forest tuning gain (tuned minus default)',
         [frame_markdown(task.table(prefix + 'rf_tuning_gain', 'tuned'))]),
        ('Top odds ratios with 95% CIs' if spec.effect_is_ratio else 'Top coefficients with 95% CIs (HC3)',
         _effects(task)),
        ('Hypothesis check', [frame_markdown(task.table(prefix + 'hypothesis_check', 'Hypothesis check'))]),
        ('Random Forest top features', _rf_top_features(task)),
        ('Drift from June to August', [frame_markdown(task.table(prefix + 'drift', 'drift'))]),
    ]
    lines = [f'## {chapter}. {spec.title}', '']
    for title, body in blocks:
        lines += [number(title), '', *body, '']
    extras = _classification_extras if spec.name == 'classification' else _regression_extras
    return lines + extras(task, root, number)


# ---------------------------------------------------------------- whole document

def what_changed(tasks):
    """The only place the previous pipeline's numbers appear."""
    cls, reg = tasks
    june_ap = cls.results[cls.results['split'] == 'June'].set_index('model')['avg_precision'] * 100
    june_mae = reg.results[reg.results['split'] == 'June'].set_index('model')['mae']
    old = PREVIOUS_PIPELINE
    new_rows = [[m, f'{june_ap[m]:.2f}'] for _, m in cls.spec.rungs if m in june_ap]
    mae_rows = [[m, f'{june_mae[m]:.2f}'] for _, m in reg.spec.rungs if m in june_mae]
    return ['## 4. What changed vs the previous pipeline', '',
            f"The previous pipeline's headline was {old['model']} with June AP {old['june_ap_percent']}% and June MAE "
            f"{old['june_mae_days']} days, chosen as the winner on June outcomes. The new pipeline selects no winner: "
            "every model is scored the same way on Train, CV, Validation, Test (June) and Monitoring (July, August), and "
            "June is the Test set. The cohorts and protocol differ, so the old numbers are context, not a like-for-like comparison.",
            '', '**New pipeline, Test (June) AP (%)**', '', md_table(['Model', 'June AP (%)'], new_rows), '',
            '**New pipeline, Test (June) MAE (days)**', '', md_table(['Model', 'June MAE (days)'], mae_rows), '']


def figure_index(root):
    names = sorted(p.name for p in (Path(root) / 'figures').glob('*.png'))
    rows = [[f'`figures/{n}`'] for n in names]
    return ['## 5. Figure index', '', md_table(['File'], rows), '']


def build_markdown(tasks, root):
    """Compose RESULTS_SUMMARY.md from both tasks' exports."""
    cls, reg = tasks
    head = ['# Results summary: simple to complex', '',
            'Generated by `python -m src.summary` from the exported tables and metadata only; nothing is refitted. '
            'Protocol: Train (in-sample), CV (5 chronological folds inside Train), Validation (30-day out-of-time holdout), '
            'Test (June) and Monitoring (July, August) scored by frozen models. No model is selected. '
            'Primary metrics: AP for classification, RMSE for regression.', '',
            '## 1. The simple-to-complex ladder', '',
            f'**{cls.spec.title}** ({cls.spec.metric_label})', '', ladder_markdown_table(cls), '',
            f'**{reg.spec.title}** ({reg.spec.metric_label})', '', ladder_markdown_table(reg), '',
            *_figure_links(root, [('04_ladder.png', 'Simple-to-complex ladder'),
                                  ('04_overfitting_gaps.png', 'Overfitting gaps'),
                                  ('04_drift_summary.png', 'Drift summary')])]
    return '\n'.join(head + task_section(cls, 2, root) + task_section(reg, 3, root)
                     + what_changed(tasks) + figure_index(root))


# ---------------------------------------------------------------- merged metadata

def merge_metadata(tasks):
    """Both tasks' run metadata plus the union of package versions (conflicts are reported)."""
    versions, conflicts = {}, []
    for task in tasks:
        for package, version in task.metadata['versions'].items():
            if versions.setdefault(package, version) != version:
                conflicts.append(package)
    return {'tasks': {task.spec.name: task.metadata for task in tasks},
            'package_versions': versions, 'package_version_conflicts': sorted(set(conflicts))}


def build_summary(root=DEFAULT_ROOT):
    """Write the 04 figures, `RESULTS_SUMMARY.md` and the merged `run_metadata.json`; return their paths."""
    root = Path(root)
    tasks = [load_task(name, root) for name in TASKS]
    fig_dir = root / 'figures'
    paths = {'ladder': fig_dir / '04_ladder.png', 'gaps': fig_dir / '04_overfitting_gaps.png',
             'drift': fig_dir / '04_drift_summary.png', 'summary': root / 'RESULTS_SUMMARY.md',
             'metadata': root / 'run_metadata.json'}
    for plot, key in ((plot_ladder, 'ladder'), (plot_overfitting_gaps, 'gaps'), (plot_drift_summary, 'drift')):
        plt.close(plot(tasks, paths[key]))
    paths['summary'].write_text(build_markdown(tasks, root), encoding='utf-8')
    paths['metadata'].write_text(json.dumps(merge_metadata(tasks), indent=2), encoding='utf-8')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default=str(DEFAULT_ROOT), help='Folder holding the task exports')
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    for key, path in build_summary(args.root).items():
        print(f'{key}: {path}')


if __name__ == '__main__':
    main()
