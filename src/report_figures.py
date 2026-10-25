"""Report figures shared by the notebooks.

Each function takes the notebook's own tables, draws one figure inline and, when
`save_path` is given, also writes it as a PNG for the report.
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948', '#c3c2b7'
UNUSED = '#efeeea'
PURPLE, SLATE = '#8060b8', '#668a9b'
ORDINAL_BLUES = ['#86b6ef', '#5598e7', '#256abf', '#104281']  # Light to dark, for ordered groups

STYLE = {
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 10, 'text.color': INK,
    'axes.labelcolor': INK2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'legend.frameon': False, 'legend.fontsize': 9.5,
}


def _style(ax, grid_axis='x'):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _titles(fig, title, subtitle, top=0.97):
    fig.text(0.02, top, title, fontsize=13, fontweight='bold', ha='left', va='top')
    fig.text(0.02, top - 0.065, subtitle, fontsize=9.5, color=INK2, ha='left', va='top')


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_waiting_period_audit(waiting_period_audit, save_path=None):
    """Share of training orders hitting the cap, grouped by cap length, one bar per run date."""
    capped = waiting_period_audit.pivot(index='waiting_period', columns='run_date', values='capped_pct')
    width = 0.76 / len(capped.columns)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.14)
        for k, run_date in enumerate(capped.columns):
            xs = [i + (k - (len(capped.columns) - 1) / 2) * width for i in range(len(capped))]
            ax.bar(xs, capped[run_date], width=width - 0.025,
                   color=ORDINAL_BLUES[k % len(ORDINAL_BLUES)], label=run_date)
        for i, (_, row) in enumerate(capped.iterrows()):  # Label only the highest bar per cap
            k = int(row.to_numpy().argmax())
            ax.text(i + (k - (len(capped.columns) - 1) / 2) * width, row.iloc[k] + 0.4,
                    f'{row.iloc[k]:.1f}%', ha='center', va='bottom', fontsize=9.5)
        ax.set_xticks(range(len(capped)), [f'{int(h)}-day cap' for h in capped.index])
        ax.set_xlabel('Waiting-period cap (days after approval)')
        ax.set_ylabel('Training orders hitting the cap (%)')
        ax.set_ylim(0, capped.to_numpy().max() * 1.18)
        _style(ax, 'y')
        ax.legend(title='Run date', ncol=len(capped.columns), loc='upper right', title_fontsize=9.5,
                  columnspacing=1.2, handlelength=1.2, bbox_to_anchor=(1.0, 1.02))
        _titles(fig, 'Waiting-period audit: orders hitting the cap',
                'Share of training orders recorded at the cap, by cap length and run date. '
                'Highest value per cap is labelled.')
        _finish(fig, save_path)


def plot_split_and_folds(train_dates, validation_dates, folds, task_label='', save_path=None):
    """Timeline of the CV folds and the Validation block inside the historical window.

    `train_dates` must be positionally aligned with the fold indices in `folds`.
    Each blue bar carries its Train order count; each orange (CV) and green (Validation) block is
    labelled above with its date range and order count.
    """
    train_dates = pd.to_datetime(pd.Series(train_dates)).dt.normalize().reset_index(drop=True)
    validation_dates = pd.to_datetime(pd.Series(validation_dates)).dt.normalize()
    one_day = pd.Timedelta(days=1)
    start = train_dates.min()
    validation_start, validation_end = validation_dates.min(), validation_dates.max() + one_day
    num, height, gap = mdates.date2num, 0.46, 0.6
    span = lambda first, last: f'{first:%d %b}\u2013{last:%d %b}'  # inclusive date range

    def block(ax, y, left, right, color, pad=0.0):
        ax.barh(y, num(right) - num(left) - pad, left=num(left) + pad, height=height, color=color)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.15)
        labels = []
        for k, (train_idx, valid_idx) in enumerate(folds, 1):
            y = len(folds) - k + 1
            train_to = train_dates.iloc[train_idx].max() + one_day
            valid_from, valid_to = train_dates.iloc[valid_idx].min(), train_dates.iloc[valid_idx].max() + one_day
            block(ax, y, start, validation_end, UNUSED)
            block(ax, y, train_dates.iloc[train_idx].min(), train_to, BLUE)
            block(ax, y, valid_from, valid_to, ORANGE, pad=gap)
            train_from = train_dates.iloc[train_idx].min()
            ax.text((num(train_from) + num(train_to)) / 2, y, f'{len(train_idx):,} Train',
                    ha='center', va='center', fontsize=8, color='white')
            ax.text((num(train_to) + num(valid_from)) / 2, y, '45d', ha='center', va='center',
                    fontsize=8, color=INK2)
            ax.text((num(valid_from) + num(valid_to)) / 2, y + height / 2 + 0.06,
                    f'{span(valid_from, valid_to - one_day)} | {len(valid_idx):,} orders',
                    ha='center', va='bottom', fontsize=8, color=INK2)
            labels.append((y, f'CV fold {k}'))
        block(ax, 0, start, validation_end, UNUSED)
        train_end = train_dates.max() + one_day
        block(ax, 0, start, train_end, BLUE)
        block(ax, 0, validation_start, validation_end, AQUA, pad=gap)
        if train_end < validation_start:
            ax.text((num(train_end) + num(validation_start)) / 2, 0, '45-day gap',
                    ha='center', va='center', fontsize=8, color=INK2)
        ax.text((num(start) + num(train_end)) / 2, 0, f'{len(train_dates):,} Train',
                ha='center', va='center', fontsize=8, color='white')
        ax.text(num(validation_end), height / 2 + 0.06,  # right-aligned so it stays inside the axes
                f'Validation {span(validation_start, validation_end - one_day)} | {len(validation_dates):,} orders',
                ha='right', va='bottom', fontsize=8, color=INK2)
        labels.append((0, 'Train and\nValidation'))
        ax.set_yticks([y for y, _ in labels], [text for _, text in labels])
        ax.set_ylim(-0.6, len(folds) + 0.75)
        ax.set_xlim(num(start) - 2, num(validation_end) + 2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if (validation_end - start).days > 240 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xlabel('Order approval date')
        _style(ax, 'x')
        ax.spines['left'].set_visible(False)
        validation_days = (validation_end - validation_start).days
        ax.legend(handles=[Patch(color=BLUE, label='Train'), Patch(color=ORANGE, label='CV validation (30 days)'),
                           Patch(color=AQUA, label=f'Validation ({validation_days} days)'),
                           Patch(color=UNUSED, label='Excluded gap')],
                  ncol=4, loc='lower left', bbox_to_anchor=(-0.02, 1.0), columnspacing=1.4, handlelength=1.2)
        _titles(fig, f'{task_label + ": " if task_label else ""}Train, CV folds and Validation',
                'Expanding folds, 45-day gap before each scored block. '
                'Test (June) and Monitoring: see 01_validation_timeline.png.')
        _finish(fig, save_path)


def plot_buffer_audit(review, target_pct=99.0, save_path=None):
    """Label completeness against buffer length, one line per run date, for each scope."""
    scopes = list(review['scope'].unique())
    colors = ORDINAL_BLUES[1:]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(scopes), figsize=(9, 4.3), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.15, wspace=0.08)
        for ax, scope in zip(axes, scopes):
            rows = review[review['scope'].eq(scope)]
            for color, (run_date, group) in zip(colors, rows.groupby('run_date')):
                ax.plot(group['buffer_days'], group['completeness_pct'], marker='o', color=color,
                        linewidth=2, label=str(run_date))
            ax.axhline(target_pct, color=RED, linewidth=1, linestyle='--')
            ax.set_xticks(sorted(rows['buffer_days'].unique()))
            ax.set_xlabel('Buffer before inference (days)')
            ax.set_title(scope.capitalize(), loc='left', fontsize=10.5, fontweight='bold', color=INK2)
            _style(ax, 'y')
        axes[0].set_ylabel('Orders with a known label (%)')
        axes[0].legend(title='Run date', loc='lower right', title_fontsize=9)
        axes[-1].text(axes[-1].get_xlim()[1], target_pct, f' {target_pct:g}% target', color=RED,
                      va='bottom', ha='right', fontsize=9)
        _titles(fig, 'Label completeness by buffer length',
                'Share of training-window orders whose late label is known on the run date.')
        _finish(fig, save_path)


def plot_skewness(reports, save_path=None):
    """Train skewness before and after log1p for the log1p candidates, one panel per task.

    `reports` maps a task name to the candidate rows of `data_audits.skewness_report`.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(reports), figsize=(10.5, 4.6), sharey=True)
        fig.subplots_adjust(left=0.27, right=0.98, top=0.74, bottom=0.14, wspace=0.08)
        for ax, (task, table) in zip(np.atleast_1d(axes), reports.items()):
            table = table.iloc[::-1]
            y = np.arange(len(table))
            ax.barh(y + 0.19, table['skew_raw'], height=0.34, color=ORANGE)
            ax.barh(y - 0.19, table['skew_log1p'], height=0.34, color=BLUE)
            for limit in (-1, 1):
                ax.axvline(limit, color=INK2, linewidth=0.8, linestyle='--')
            ax.axvline(0, color=INK2, linewidth=0.8)
            ax.set_yticks(y, table['feature'])
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xticks([-1, 0, 1, 3, 10], ['-1', '0', '1', '3', '10'])
            ax.set_xlabel('Skewness on Train (symmetric log axis)')
            ax.set_title(task.capitalize(), loc='left', fontsize=10.5, fontweight='bold', color=INK2)
            _style(ax, 'x')
        fig.legend(
            handles=[Patch(color=ORANGE, label='Raw'), Patch(color=BLUE, label='After log1p'),
                     Line2D([0], [0], color=INK2, linestyle='--', linewidth=0.8, label='|skew| = 1')],
            ncol=3, loc='lower left', bbox_to_anchor=(0.02, 0.80), columnspacing=1.2, handlelength=1.4)
        _titles(fig, 'Skewness before and after log1p',
                'The eight log1p candidates; the rule is to transform when the raw |skew| exceeds 1.')
        _finish(fig, save_path)


def plot_cyclic_calendar(save_path=None):
    """Why month is encoded as a sine/cosine pair: December sits next to January on a circle."""
    months = np.arange(1, 13)
    angle = 2 * np.pi * months / 12
    names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    with plt.rc_context(STYLE):
        fig, (ax_circle, ax_wave) = plt.subplots(1, 2, figsize=(10, 4.4),
                                                 gridspec_kw={'width_ratios': [1, 1.25]})
        fig.subplots_adjust(left=0.05, right=0.98, top=0.74, bottom=0.15, wspace=0.18)
        ax_circle.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)),
                       color=GRID, linewidth=1.5)
        colors = [RED if m in (1, 12) else BLUE for m in months]
        ax_circle.scatter(np.cos(angle), np.sin(angle), s=70, color=colors, zorder=3)
        for m, a, name in zip(months, angle, names):
            ax_circle.text(1.2 * np.cos(a), 1.2 * np.sin(a), name, ha='center', va='center', fontsize=9,
                           color=RED if m in (1, 12) else INK2)
        ax_circle.set_xlim(-1.45, 1.45)
        ax_circle.set_ylim(-1.45, 1.45)
        ax_circle.set_aspect('equal')
        ax_circle.set_xlabel('cosine of month angle')
        ax_circle.set_ylabel('sine of month angle')
        ax_circle.set_title('Months on a circle', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
        _style(ax_circle, 'both')
        grid = np.linspace(1, 12, 200)
        ax_wave.plot(grid, np.sin(2 * np.pi * grid / 12), color=BLUE, linewidth=2, label='sine')
        ax_wave.plot(grid, np.cos(2 * np.pi * grid / 12), color=ORANGE, linewidth=2, label='cosine')
        ax_wave.scatter(months, np.sin(angle), color=BLUE, s=22, zorder=3)
        ax_wave.scatter(months, np.cos(angle), color=ORANGE, s=22, zorder=3)
        ax_wave.set_xticks(months, names)
        ax_wave.set_xlabel('Order approval month')
        ax_wave.set_ylabel('Encoded value (unitless)')
        ax_wave.set_title('Sine and cosine of the month', loc='left', fontsize=10.5, fontweight='bold',
                          color=INK2)
        ax_wave.legend(loc='lower left', ncol=2)
        _style(ax_wave, 'y')
        gap_raw, gap_circle = 11, 2 * np.sin(np.pi / 12)
        _titles(fig, 'Cyclic calendar encoding for the linear models',
                f'As a number, December and January are {gap_raw} apart; on the circle they are '
                f'{gap_circle:.2f} apart, like any other neighbouring months.')
        _finish(fig, save_path)
