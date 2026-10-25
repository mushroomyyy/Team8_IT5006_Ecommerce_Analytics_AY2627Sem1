"""Report figures for the regression track.

Each function takes the notebook's own tables, draws one figure inline and, when
`save_path` is given, also writes it as a PNG for the report.
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948', '#c3c2b7'
UNUSED = '#efeeea'
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


def _model_label(model, variant):
    return f"{model} ({'reference baseline' if variant == 'reference' else variant})"


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
        ax.set_ylabel('Training orders hitting the cap (%)')
        ax.set_ylim(0, capped.to_numpy().max() * 1.18)
        _style(ax, 'y')
        ax.legend(title='Run date', ncol=len(capped.columns), loc='upper right', title_fontsize=9.5,
                  columnspacing=1.2, handlelength=1.2, bbox_to_anchor=(1.0, 1.02))
        _titles(fig, 'Waiting-period audit: orders hitting the cap',
                'Share of training orders recorded at the cap, by cap length and run date. '
                'Highest value per cap is labelled.')
        _finish(fig, save_path)


def plot_split_and_folds(train_dates, test_dates, folds, save_path=None):
    """Timeline of historical CV folds and the development holdout.

    `train_dates` must be positionally aligned with the fold indices in `folds`.
    """
    train_dates = pd.to_datetime(pd.Series(train_dates)).dt.normalize().reset_index(drop=True)
    test_dates = pd.to_datetime(pd.Series(test_dates)).dt.normalize()
    one_day = pd.Timedelta(days=1)
    start, test_start, test_end = train_dates.min(), test_dates.min(), test_dates.max() + one_day
    num, height, gap = mdates.date2num, 0.46, 0.6

    def block(ax, y, left, right, color, pad=0.0):
        ax.barh(y, num(right) - num(left) - pad, left=num(left) + pad, height=height, color=color)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.17)
        labels = []
        for k, (train_idx, valid_idx) in enumerate(folds, 1):
            y = len(folds) - k + 1
            train_from = train_dates.iloc[train_idx].min()
            train_to = train_dates.iloc[train_idx].max() + one_day
            valid_from, valid_to = train_dates.iloc[valid_idx].min(), train_dates.iloc[valid_idx].max() + one_day
            block(ax, y, start, test_end, UNUSED)
            block(ax, y, train_from, train_to, BLUE)
            block(ax, y, valid_from, valid_to, ORANGE, pad=gap)
            ax.text((num(valid_from) + num(valid_to)) / 2, y + height / 2 + 0.06, f'{len(valid_idx):,}',
                    ha='center', va='bottom', fontsize=8.5, color=INK2)
            labels.append((y, f'CV fold {k}'))
        block(ax, 0, start, test_end, UNUSED)
        train_end = train_dates.max() + one_day
        block(ax, 0, start, train_end, BLUE)
        block(ax, 0, test_start, test_end, AQUA, pad=gap)
        if train_end < test_start:
            ax.text((num(train_end) + num(test_start)) / 2, 0, '45-day gap',
                    ha='center', va='center', fontsize=8, color=INK2)
        ax.text((num(test_start) + num(test_end)) / 2, height / 2 + 0.06, f'{len(test_dates):,}',
                ha='center', va='bottom', fontsize=8.5, color=INK2)
        labels.append((0, 'Development\nholdout split'))
        ax.set_yticks([y for y, _ in labels], [text for _, text in labels])
        ax.set_ylim(-0.6, len(folds) + 0.75)
        ax.set_xlim(num(start) - 2, num(test_end) + 2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if (test_end - start).days > 240 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        _style(ax, 'x')
        ax.spines['left'].set_visible(False)
        test_days = (test_end - test_start).days
        ax.legend(handles=[Patch(color=BLUE, label='Train'), Patch(color=ORANGE, label='CV validation'),
                           Patch(color=AQUA, label=f'Development holdout ({test_days} days)'),
                           Patch(color=UNUSED, label='Excluded')],
                  ncol=4, loc='lower left', bbox_to_anchor=(-0.02, 1.0), columnspacing=1.4, handlelength=1.2)
        _titles(fig, 'Historical training, CV and development holdout',
                'Historical data only. Grey gaps separate training from CV validation; numbers are order counts.')
        fig.text(0.04, 0.025,
                 'The green development holdout is separate from the later prediction cohort.',
                 fontsize=8.5, color=INK2)
        _finish(fig, save_path)


def plot_cv_mae(cv_results, selected_key, save_path=None):
    """CV MAE with ±1 std for every model and variant; the supplied selected candidate is highlighted."""
    table = cv_results.sort_values('cv_mae_mean', ascending=False)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(4.2, 0.42 * len(table) + 1.8)))
        fig.subplots_adjust(left=0.29, right=0.95, top=0.74, bottom=0.11)
        for i, (key, row) in enumerate(table.iterrows()):
            is_selected = key == selected_key
            ax.barh(i, row['cv_mae_mean'], height=0.52, color=BLUE if is_selected else GRAY)
            ax.errorbar(row['cv_mae_mean'], i, xerr=row['cv_mae_std'], color=INK2, linewidth=1, capsize=2.5)
            ax.text(row['cv_mae_mean'] + row['cv_mae_std'] + 0.2, i, f"{row['cv_mae_mean']:.2f}",
                    va='center', fontsize=9.5, fontweight='bold' if is_selected else 'normal')
        ax.set_yticks(range(len(table)), [_model_label(*key) for key in table.index])
        ax.set_xlabel('Cross-validated MAE (days); lower is better')
        ax.set_xlim(0, (table['cv_mae_mean'] + table['cv_mae_std']).max() * 1.12)
        _style(ax, 'x')
        ax.legend(handles=[Patch(color=BLUE, label='Selected candidate'), Patch(color=GRAY, label='Other models'),
                           Line2D([0], [0], color=INK2, linewidth=1, label='± 1 std across the folds')],
                  ncol=3, loc='lower left', bbox_to_anchor=(-0.42, 1.0), columnspacing=1.4, handlelength=1.2)
        _titles(fig, 'Cross-validated MAE, default vs tuned',
                'Time-series CV on the training part (expanding-window folds). '
                f'Selected candidate: {selected_key[0]} ({selected_key[1]}).', top=0.975)
        _finish(fig, save_path)


def plot_mae_across_evaluations(cv_results, holdout_results, backtest_pooled, selected_key, save_path=None):
    """One row per model: its MAE under CV, on the hold-out and pooled over the backtest."""
    table = pd.DataFrame({'cv': cv_results['cv_mae_mean'], 'holdout': holdout_results['mae'],
                          'backtest': backtest_pooled['mae']}).dropna().sort_values('backtest', ascending=False)
    series = [('cv', 'Cross-validation', BLUE, 'o', 0.17), ('holdout', 'Hold-out', ORANGE, 's', 0.0),
              ('backtest', 'Prediction cohort, pooled', AQUA, 'D', -0.17)]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(4.2, 0.42 * len(table) + 1.8)))
        fig.subplots_adjust(left=0.29, right=0.97, top=0.74, bottom=0.11)
        for i, (key, row) in enumerate(table.iterrows()):
            ax.plot([row.min(), row.max()], [i, i], color=AXIS, linewidth=1.2, zorder=1)
            for column, _, color, marker, offset in series:  # Small offsets keep equal values visible
                ax.scatter(row[column], i + offset, s=58, color=color, marker=marker,
                           edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.set_yticks(range(len(table)), [_model_label(*key) for key in table.index])
        if selected_key in table.index:
            ax.get_yticklabels()[table.index.get_loc(selected_key)].set_fontweight('bold')
        ax.set_xlabel('MAE (days); lower is better')
        ax.set_ylim(-0.6, len(table) - 0.4)
        _style(ax, 'x')
        ax.legend(handles=[Line2D([0], [0], marker=marker, color='none', markerfacecolor=color,
                                  markeredgecolor=SURFACE, markersize=8, label=label)
                           for _, label, color, marker, _ in series],
                  ncol=3, loc='lower left', bbox_to_anchor=(-0.42, 1.0), columnspacing=1.2, handletextpad=0.3)
        _titles(fig, 'MAE under cross-validation, hold-out and backtest',
                'Sorted by backtest MAE; the selected candidate is bold.', top=0.975)
        _finish(fig, save_path)


def plot_backtest_by_month(backtest_monthly, series, save_path=None):
    """Monthly backtest MAE as lines. `series` is a list of (model, variant, label), at most four."""
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    months = sorted(backtest_monthly['month'].unique())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        fig.subplots_adjust(left=0.08, right=0.70, top=0.76, bottom=0.11)
        ends, top = [], 0
        for (model, variant, label), color in zip(series, colors):
            rows = backtest_monthly[backtest_monthly['model'].eq(model) & backtest_monthly['variant'].eq(variant)]
            values = rows.set_index('month')['mae'].reindex(months).to_numpy()
            ax.plot(range(len(months)), values, color=color, linewidth=2, solid_capstyle='round', label=label)
            ax.scatter([len(months) - 1], [values[-1]], s=52, color=color, edgecolor=SURFACE,
                       linewidth=1.4, zorder=3)
            ends.append((values[-1], f'{label}  {values[-1]:.1f}'))
            top = max(top, values.max())
        # End labels: keep them a minimum distance apart and tie each to its line with a leader
        min_gap, position = top * 0.075, None
        for value, text in sorted(ends):
            position = value - min_gap * 0.6 if position is None else max(value, position + min_gap)
            ax.annotate(text, xy=(len(months) - 1, value), xytext=(len(months) - 1 + 0.2, position),
                        fontsize=9.5, va='center', annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color=AXIS, linewidth=0.8))
        ax.set_xticks(range(len(months)), [pd.Period(m).strftime('%b %Y') for m in months])
        ax.set_xlim(-0.2, len(months) - 0.85)
        ax.set_ylim(0, top * 1.08)
        ax.set_ylabel('MAE (days)')
        _style(ax, 'y')
        ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(-0.09, 1.0), columnspacing=1.3, handlelength=1.4)
        _titles(fig, 'MAE by evaluation month',
                'Monthly approval cohorts; fitting policy is defined by the caller. '
                'End values are the last month.')
        _finish(fig, save_path)


def plot_feature_importance(importance, model_label, save_path=None, cohort_label="hold-out"):
    """Permutation importance as bars; `importance` has feature and mae_increase_days columns."""
    table = importance.sort_values('mae_increase_days')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(table) + 0.9))
        fig.subplots_adjust(left=0.32, right=0.95, top=0.80, bottom=0.13)
        ax.barh(range(len(table)), table['mae_increase_days'], height=0.5, color=BLUE)
        limit = table['mae_increase_days'].max() * 1.15
        for i, value in enumerate(table['mae_increase_days']):
            ax.text(value + limit * 0.015, i, f'{value:.2f}', va='center', fontsize=9.5)
        ax.set_yticks(range(len(table)), table['feature'])
        ax.set_xlabel(f'Increase in {cohort_label} MAE when the feature is shuffled (days)')
        ax.set_xlim(0, limit)
        _style(ax, 'x')
        _titles(fig, f'Top {len(table)} features by permutation importance',
                f'{model_label} on the {cohort_label} cohort. Longer bars mean the model relies on the feature more.',
                top=0.975)
        _finish(fig, save_path)


def plot_residuals_by_state(residuals, model_label, top_n=12, save_path=None, cohort_label="Hold-out"):
    """Bias and MAE by customer state for the `top_n` states with the most hold-out orders."""
    table = residuals.groupby('customer_state').agg(
        orders=('error', 'size'), bias=('error', 'mean'), mae=('error', lambda e: e.abs().mean()))
    table = table.sort_values('orders', ascending=False).head(top_n).iloc[::-1]
    with plt.rc_context(STYLE):
        fig, (ax_bias, ax_mae) = plt.subplots(1, 2, figsize=(9, 0.33 * len(table) + 1.1), sharey=True)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.12, wspace=0.08)
        ax_bias.barh(range(len(table)), table['bias'], height=0.5,
                     color=[RED if bias > 0 else BLUE for bias in table['bias']])
        span = table['bias'].abs().max()
        for i, bias in enumerate(table['bias']):
            ax_bias.text(bias + (0.035 if bias > 0 else -0.035) * span, i, f'{bias:+.1f}', va='center',
                         ha='left' if bias > 0 else 'right', fontsize=9)
        ax_bias.axvline(0, color=INK2, linewidth=0.8)
        ax_bias.set_xlim(min(table['bias'].min(), 0) - 0.3 * span, max(table['bias'].max(), 0) + 0.3 * span)
        ax_bias.set_xlabel('Bias: predicted − actual (days)')
        ax_bias.set_yticks(range(len(table)), [f'{state}  ({orders:,})'
                                               for state, orders in zip(table.index, table['orders'])])
        ax_bias.set_title('Bias', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
        _style(ax_bias, 'x')
        ax_bias.legend(handles=[Patch(color=BLUE, label='Under-predicts'), Patch(color=RED, label='Over-predicts')],
                       ncol=2, loc='lower left', bbox_to_anchor=(0.18, 1.0), columnspacing=1.2, handlelength=1.2)
        ax_mae.barh(range(len(table)), table['mae'], height=0.5, color=GRAY)
        for i, mae in enumerate(table['mae']):
            ax_mae.text(mae + table['mae'].max() * 0.015, i, f'{mae:.1f}', va='center', fontsize=9)
        ax_mae.set_xlim(0, table['mae'].max() * 1.16)
        ax_mae.set_xlabel('MAE (days)')
        ax_mae.set_title('Typical error size', loc='left', fontsize=10.5, fontweight='bold', color=INK2)
        _style(ax_mae, 'x')
        ax_mae.spines['left'].set_visible(False)
        _titles(fig, f'{cohort_label} residuals by customer state',
                f'{model_label}, the {len(table)} customer states with the most {cohort_label.lower()} orders '
                '(order count in brackets).', top=0.975)
        _finish(fig, save_path)
