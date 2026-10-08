"""Shared model-result figures for the classification and regression notebooks.

Each function takes plain tables (see `src.evaluation`), draws one figure with a title,
labelled axes and a legend, saves it as a 200-dpi PNG when `save_path` is given, and
returns the figure. Model colours are fixed so a model looks the same in every figure.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from . import report_figures as figs

MODEL_COLORS = {'C0': figs.GRAY, 'R0': figs.GRAY, 'C1': figs.BLUE, 'R1': figs.BLUE,
                'C2': figs.AQUA, 'R2': figs.AQUA, 'C2B': figs.YELLOW, 'R2B': figs.YELLOW,
                'C3': figs.ORANGE, 'R3': figs.ORANGE, 'C3d': '#8e5bd1', 'R3d': '#8e5bd1'}
MODEL_MARKERS = {'C3d': 's', 'R3d': 's', 'C0': 'x', 'R0': 'x'}


def _color(model):
    return MODEL_COLORS.get(str(model).split(':')[0].split(' ')[0], figs.INK2)


def _colors_for(labels):
    """Fixed colour for model keys; other labels (e.g. resampling options) cycle the palette."""
    cycle = iter([figs.BLUE, figs.ORANGE, figs.AQUA, figs.YELLOW, figs.RED])
    return {label: (_color(label) if str(label).split(':')[0].split(' ')[0] in MODEL_COLORS else next(cycle))
            for label in labels}


def _finish(fig, title, save_path, subtitle=None):
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, fontweight='bold')
    if subtitle:
        fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=figs.INK2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_stepwise_path(path, chosen_step, best_step, ylabel, save_path=None, title=None):
    """CV score against stepwise step with +/-1 SE band, the best step and the 1-SE choice marked."""
    with plt.rc_context(figs.STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(path['step'], path['cv_mean'] - path['cv_se'], path['cv_mean'] + path['cv_se'],
                        color=figs.BLUE, alpha=0.18, label='CV mean +/- 1 SE (5 folds)')
        ax.plot(path['step'], path['cv_mean'], color=figs.BLUE, marker='o', ms=4, label='CV mean')
        best = path.loc[path['step'] == best_step].iloc[0]
        ax.axhline(best['cv_mean'] - best['cv_se'], color=figs.GRAY, ls='--', lw=1,
                   label='Best step minus 1 SE')
        ax.scatter([best_step], [best['cv_mean']], s=90, color=figs.ORANGE, zorder=3,
                   label=f'Best CV step ({best_step})')
        chosen = path.loc[path['step'] == chosen_step].iloc[0]
        ax.scatter([chosen_step], [chosen['cv_mean']], s=140, facecolor='none', edgecolor=figs.RED,
                   linewidth=2, zorder=4, label=f'1-SE choice (step {chosen_step})')
        ax.set_xticks(path['step'], [f"{r.step}: {r.added_unit}" if r.step else '0' for r in path.itertuples()],
                      rotation=90, fontsize=8)
        ax.set(xlabel='Step and the raw feature added at that step', ylabel=ylabel)
        ax.legend(loc='lower right')
        figs._style(ax, 'y')
        return _finish(fig, title or 'Forward stepwise selection judged by chronological CV', save_path)


def plot_decile_lift(deciles, save_path=None, title='June risk deciles: capture and lift'):
    """Cumulative capture and per-decile lift for each model; `deciles` maps model -> risk_decile_coverage."""
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].plot([0, 10], [0, 100], color=figs.GRAY, ls='--', lw=1, label='Random ranking')
        for model, table in deciles.items():
            x = np.r_[0, table['decile']]
            axes[0].plot(x, np.r_[0, table['cumulative_capture_pct']], marker=MODEL_MARKERS.get(model, 'o'),
                         ms=4, color=_color(model), label=model)
            axes[1].plot(table['decile'], table['lift'], marker=MODEL_MARKERS.get(model, 'o'), ms=4,
                         color=_color(model), label=model)
        axes[1].axhline(1, color=figs.GRAY, ls='--', lw=1, label='No lift (1.0)')
        axes[0].set(xlabel='Risk decile (1 = highest predicted risk, cumulative)',
                    ylabel='Late orders captured (%)', xticks=range(0, 11))
        axes[1].set(xlabel='Risk decile (1 = highest predicted risk)',
                    ylabel='Lift (late rate in decile / overall late rate)', xticks=range(1, 11))
        for ax in axes:
            figs._style(ax, 'y')
            ax.legend()
        return _finish(fig, title, save_path)


def plot_model_comparison(values, ylabel, sds=None, save_path=None, title='Model comparison',
                          subtitle=None):
    """Dot chart of a metric by split (columns of `values`) for each model (rows).

    `sds` (same shape) adds error bars, used for the CV column.
    """
    with plt.rc_context(figs.STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        n = len(values)
        for i, (model, row) in enumerate(values.iterrows()):
            x = np.arange(len(row)) + (i - (n - 1) / 2) * 0.1
            error = None if sds is None else sds.loc[model].fillna(0).to_numpy()
            ax.errorbar(x, row.to_numpy(), yerr=error, fmt=MODEL_MARKERS.get(model, 'o'),
                        color=_color(model), ms=6, capsize=3, label=model)
        ax.set_xticks(range(len(values.columns)), values.columns)
        ax.set(xlabel='Evaluation split', ylabel=ylabel)
        ax.legend(ncol=2)
        figs._style(ax, 'y')
        return _finish(fig, title, save_path, subtitle)


def plot_drift(drift, metric, ylabel, periods=('June', 'July', 'August'), save_path=None, title=None):
    """Line per model across the Test (June) and Monitoring months from `evaluation.drift_table`."""
    frame = drift[drift['metric'] == metric]
    with plt.rc_context(figs.STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for _, row in frame.iterrows():
            ax.plot(list(periods), [row[p] for p in periods], marker=MODEL_MARKERS.get(row['model'], 'o'),
                    color=_color(row['model']), label=row['model'])
        ax.set(xlabel='Month (June = Test, July and August = Monitoring)', ylabel=ylabel)
        ax.legend(ncol=2)
        figs._style(ax, 'y')
        return _finish(fig, title or f'{ylabel} from June to August (frozen models)', save_path)


def plot_calibration(panels, save_path=None, n_bins=10, title='Calibration', xlabel=None):
    """Reliability curves. `panels` maps a panel title to {label: (y_true, probability)}.

    Bins hold equal numbers of orders. Points on the diagonal are well calibrated.
    """
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
        for ax, (name, curves) in zip(axes[0], panels.items()):
            top = 0.0
            colors = _colors_for(curves)
            for label, (y_true, prob) in curves.items():
                observed, predicted = calibration_curve(y_true, prob, n_bins=n_bins, strategy='quantile')
                top = max(top, predicted.max(), observed.max())
                ax.plot(predicted, observed, marker=MODEL_MARKERS.get(label, 'o'), ms=4,
                        color=colors[label], label=label)
            top = min(1.0, top * 1.1)
            ax.plot([0, top], [0, top], color=figs.GRAY, ls='--', lw=1, label='Perfect calibration')
            ax.set(title=name, xlabel=xlabel or 'Mean predicted probability of late delivery',
                   ylabel='Observed share of late orders', xlim=(0, top), ylim=(0, top))
            figs._style(ax, 'both')
            ax.legend()
        return _finish(fig, title, save_path)


def plot_effects_forest(table, effect_label, save_path=None, title=None, top=15):
    """Forest plot of per-1-SD effects with 95% CIs from `interpretation.coefficient_table`.

    Odds ratios (log axis) for classification, days for regression. Terms with huge
    standard errors (not identified) should be removed first.
    """
    data = table[table['effect_per_sd'].notna()].copy()
    data['strength'] = (data['coef'] / data['se']).abs()
    data = data.nlargest(top, 'strength').iloc[::-1]
    odds = data['effect_type'].iloc[0] == 'odds ratio'
    with plt.rc_context(figs.STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.36 * len(data) + 1.6))
        y = np.arange(len(data))
        ax.errorbar(data['effect_per_sd'], y, xerr=[data['effect_per_sd'] - data['effect_per_sd_low'],
                    data['effect_per_sd_high'] - data['effect_per_sd']], fmt='o', color=figs.BLUE, capsize=3)
        ax.axvline(1 if odds else 0, color=figs.GRAY, ls='--', lw=1)
        if odds:
            ax.set_xscale('log')
        ax.set_yticks(y, data['term'])
        ax.set(xlabel=effect_label, ylabel='Model term')
        figs._style(ax, 'x')
        return _finish(fig, title or 'Effects per 1 SD with 95% confidence intervals', save_path)


def plot_importance(table, xlabel, save_path=None, title='Permutation importance', top=12):
    """Horizontal bars of permutation importance (mean +/- SD over repeats) for the top features."""
    data = table.head(top).iloc[::-1]
    with plt.rc_context(figs.STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(data) + 1.6))
        ax.barh(data['feature'], data['importance_mean'], xerr=data['importance_sd'], color=figs.BLUE,
                height=0.55, capsize=2)
        ax.set(xlabel=xlabel, ylabel='Raw feature')
        figs._style(ax, 'x')
        return _finish(fig, title, save_path)


def plot_resampling_comparison(summary, labels, save_path=None, title='Class-imbalance handling'):
    """CV mean +/- SD of AP, ROC-AUC, Brier and top-10% precision by resampling strategy and model."""
    metrics = [('avg_precision', 'CV average precision'), ('roc_auc', 'CV ROC-AUC'),
               ('brier', 'CV Brier score (lower is better)'), ('precision_top10', 'CV top-10% precision')]
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
        strategies = list(dict.fromkeys(summary['strategy']))
        for ax, (metric, label) in zip(axes, metrics):
            for i, model in enumerate(dict.fromkeys(summary['model'])):
                rows = summary[summary['model'] == model].set_index('strategy').loc[strategies]
                x = np.arange(len(strategies)) + (i - 0.5) * 0.2
                ax.errorbar(x, rows[f'cv_{metric}_mean'], yerr=rows[f'cv_{metric}_std'], fmt='o',
                            color=_color(model), capsize=3, label=model)
            ax.set_xticks(range(len(strategies)), [labels[s] for s in strategies], rotation=30, ha='right')
            ax.set(ylabel=label, xlabel='Resampling inside each training fold')
            figs._style(ax, 'y')
        axes[0].legend()
        return _finish(fig, title, save_path)


def partial_residual_bins(result, design, y, term, n_bins=20):
    """Binned component-plus-residual for a logistic fit: mean x, mean partial residual, fitted line.

    The partial residual is the working residual (y - p) / (p (1 - p)) plus coef * x, so a
    straight line through the bins means the term is linear on the log-odds scale.
    """
    exog = design.assign(const=1.0)[list(result.params.index)]
    p = np.clip(np.asarray(result.predict(exog)), 1e-9, 1 - 1e-9)
    working = (np.asarray(y, dtype=float) - p) / (p * (1 - p))
    x = design[term].to_numpy()
    frame = pd.DataFrame({'x': x, 'partial': working + result.params[term] * x})
    frame['bin'] = pd.qcut(frame['x'], n_bins, duplicates='drop')
    bins = frame.groupby('bin', observed=True).agg(x=('x', 'mean'), partial=('partial', 'mean')).reset_index(drop=True)
    bins['line'] = result.params[term] * bins['x'] + (bins['partial'] - result.params[term] * bins['x']).mean()
    return bins


def plot_partial_residuals(result, design, y, terms, save_path=None,
                           title='Curvature check: binned partial residuals (Train)'):
    """One panel per term; dots are equal-count bins, the line is the model's linear fit."""
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(terms), figsize=(4.6 * len(terms), 4.2), squeeze=False)
        for ax, term in zip(axes[0], terms):
            bins = partial_residual_bins(result, design, y, term)
            ax.scatter(bins['x'], bins['partial'], color=figs.BLUE, s=24, label='Bin mean (20 bins)')
            ax.plot(bins['x'], bins['line'], color=figs.ORANGE, label='Linear fit')
            ax.set(xlabel=f'{term} (standardised)', ylabel='Partial residual (log-odds)')
            figs._style(ax, 'y')
            ax.legend()
        return _finish(fig, title, save_path)


def plot_pr_curves(panels, save_path=None, top_frac=0.10, title='Precision-recall curves'):
    """Precision-recall curves per panel with AP in the legend, a no-skill line and the top-10% point.

    `panels` maps a panel title to {label: (y_true, probability)}. The no-skill line is the
    panel's late rate. The marker on each curve is the operating point that flags the
    highest-scored `top_frac` of orders (the operational capacity cut-off).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8), squeeze=False)
        for ax, (name, curves) in zip(axes[0], panels.items()):
            for label, (y_true, prob) in curves.items():
                y, p = np.asarray(y_true, dtype=int), np.asarray(prob, dtype=float)
                precision, recall, _ = precision_recall_curve(y, p)
                ax.plot(recall, precision, color=_color(label), lw=1.6,
                        label=f'{label} (AP {average_precision_score(y, p):.3f})')
                cutoff = np.sort(p)[::-1][max(1, int(np.ceil(len(p) * top_frac))) - 1]
                flagged = p >= cutoff
                ax.scatter([y[flagged].sum() / y.sum()], [y[flagged].mean()], s=60, color=_color(label),
                           edgecolor='black', zorder=4)
                late_rate = y.mean()
            ax.axhline(late_rate, color=figs.GRAY, ls='--', lw=1, label=f'No skill (late rate {late_rate:.3f})')
            ax.scatter([], [], s=60, color='white', edgecolor='black', label=f'Top {top_frac:.0%} operating point')
            ax.set(title=name, xlabel='Recall (share of late orders found)', ylabel='Precision', xlim=(0, 1))
            figs._style(ax, 'both')
            ax.legend(loc='upper right')
        return _finish(fig, title, save_path)
