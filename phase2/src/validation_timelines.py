"""Plot the June validation workflow from the rows and CV splits used by models."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


COLORS = {
    'train': '#2878D0',
    'buffer': '#E5E3DF',
    'validation': '#F36B32',
    'holdout': '#13A87A',
    'evaluation': '#8060B8',
}


def _period(rows, date_col='order_approved_dt'):
    if rows.empty:
        raise ValueError('Timeline periods must contain at least one eligible order')
    dates = pd.to_datetime(rows[date_col]).dt.normalize()
    return dates.min(), dates.max(), len(rows)


def _bar(ax, start, end, y, color, text=None, text_color='#202020', height=0.58,
         edgecolor='white'):
    """Draw an inclusive calendar-date interval as a horizontal bar."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    width = max((end - start).days + 1, 1)
    ax.barh(y, width, left=mdates.date2num(start), height=height, color=color,
            edgecolor=edgecolor, linewidth=0.6, align='center')
    if text:
        ax.text(mdates.date2num(start) + width / 2, y, text, ha='center', va='center',
                fontsize=8, color=text_color, clip_on=True)


def _draw_panel(ax, name, timeline, n_splits, validation_days, gap_days,
                holdout_gap_days, test_days, inference_start, inference_end):
    train = timeline['development_train']
    holdout = timeline['development_holdout']
    final_refit = timeline['final_refit']
    june_scored = timeline['june_scored']
    june_evaluated = timeline['june_evaluated']
    folds = timeline['cv_folds']
    date_col = timeline.get('date_col', 'order_approved_dt')

    hold_start, hold_end, hold_count = _period(holdout, date_col)
    full_start, full_end, full_count = _period(final_refit, date_col)
    _, _, score_count = _period(june_scored, date_col)

    labels = [f'CV fold {i}' for i in range(1, n_splits + 1)] + [
        'Development\nholdout', 'Final refit', 'June evaluation']
    ys = list(range(len(labels) - 1, -1, -1))
    ax.set_yticks(ys, labels)
    ax.set_title(name, loc='left', fontsize=13, fontweight='bold', pad=11)

    # Validation blocks are exact calendar intervals; their order counts come
    # from each notebook's post-eligibility dataframe and returned fold indices.
    fold_anchor = pd.to_datetime(train[date_col]).dt.normalize().max()
    first_valid = fold_anchor - pd.Timedelta(days=n_splits * validation_days - 1)
    for i, (train_idx, valid_idx) in enumerate(folds):
        y = ys[i]
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[valid_idx]
        train_start, train_end, train_count = _period(fold_train, date_col)
        valid_start = first_valid + pd.Timedelta(days=i * validation_days)
        valid_end = valid_start + pd.Timedelta(days=validation_days - 1)
        _bar(ax, train_start, train_end, y, COLORS['train'],
             f'{train_count:,} train', 'white')
        gap_start = valid_start - pd.Timedelta(days=gap_days)
        _bar(ax, gap_start, valid_start - pd.Timedelta(days=1), y,
             COLORS['buffer'], f'{gap_days}d')
        _bar(ax, valid_start, valid_end, y, COLORS['validation'])
        label_y = y + (0.36 if i % 2 == 0 else 0.55)
        ax.text(mdates.date2num(valid_start + (valid_end - valid_start) / 2), label_y,
                f'{valid_start:%d %b}–{valid_end:%d %b} | '
                f'{validation_days}d | {len(fold_valid):,} orders',
                ha='center', va='bottom', fontsize=7.2, color='#202020', clip_on=False)

    # Green development holdout and its preceding excluded buffer.
    y = ys[n_splits]
    train_start, train_end, train_count = _period(train, date_col)
    hold_gap_start = hold_start - pd.Timedelta(days=holdout_gap_days)
    _bar(ax, train_start, train_end, y, COLORS['train'],
         f'{train_count:,} train | {len(pd.date_range(train_start, train_end))} calendar days', 'white')
    _bar(ax, hold_gap_start, hold_start - pd.Timedelta(days=1), y,
         COLORS['buffer'], f'{holdout_gap_days}d')
    _bar(ax, hold_start, hold_end, y, COLORS['holdout'])
    ax.text(mdates.date2num(hold_start + (hold_end - hold_start) / 2), y + 0.34,
            f'{hold_start:%d %b}–{hold_end:%d %b} | {test_days}d | {hold_count:,} orders',
            ha='left', va='bottom', fontsize=7.5, color='#202020', clip_on=False)

    # Final model is refit on the full eligible history at the June run date.
    y = ys[n_splits + 1]
    history_days = (pd.Timestamp(timeline['window_end']) -
                    pd.Timestamp(timeline['window_start'])).days + 1
    _bar(ax, timeline['window_start'], timeline['window_end'], y, COLORS['train'],
         f'Full-history refit: {history_days} days | {full_count:,} orders', 'white')
    buffer_start = pd.Timestamp(timeline['window_end']) + pd.Timedelta(days=1)
    buffer_end = pd.Timestamp(inference_start) - pd.Timedelta(days=1)
    if buffer_start <= buffer_end:
        _bar(ax, buffer_start, buffer_end, y, COLORS['buffer'], '45d cutoff')

    # Purple bar is the scored June cohort; report how many had mature labels.
    y = ys[n_splits + 2]
    _bar(ax, inference_start, inference_end, y, COLORS['evaluation'])
    june_start = pd.Timestamp(inference_start)
    june_end = pd.Timestamp(inference_end)
    ax.text(mdates.date2num(june_start) - 3, y,
            f'{score_count:,} scored / {june_evaluated:,} evaluated; labels at approval + 45 days',
            ha='right', va='center', fontsize=8, color='#202020', clip_on=False)
    ax.text(mdates.date2num(june_end), y + 0.32,
            f'1–30 Jun | 30 days', ha='right', va='bottom', fontsize=8,
            color='#202020', clip_on=False)

    ax.grid(axis='x', color='#e8e8e8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(pd.Timestamp(timeline['window_start']) - pd.Timedelta(days=8),
                pd.Timestamp(inference_end) + pd.Timedelta(days=10))


def plot_validation_timelines(classification, regression, output_path,
                              n_splits=5, validation_days=30, gap_days=45,
                              holdout_gap_days=45, test_days=30,
                              inference_start='2018-06-01', inference_end='2018-06-30'):
    """Save the two-panel June timeline using actual eligible rows and fold indices.

    Each timeline mapping must provide `development_train`, `development_holdout`,
    `cv_folds`, `final_refit`, `june_scored`, `june_evaluated`, `window_start`, and
    `window_end`. Fold indices must be the exact post-availability indices used in
    the corresponding model notebook.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.42})
    _draw_panel(axes[0], 'Classification — June 2018 run', classification,
                n_splits, validation_days, gap_days, holdout_gap_days,
                test_days, inference_start, inference_end)
    _draw_panel(axes[1], 'Regression — June 2018 run', regression,
                n_splits, validation_days, gap_days, holdout_gap_days,
                test_days, inference_start, inference_end)

    legend = [
        patches.Patch(color=COLORS['train'], label='Training / refit'),
        patches.Patch(color=COLORS['buffer'], label='Excluded buffer'),
        patches.Patch(color=COLORS['validation'], label='CV validation'),
        patches.Patch(color=COLORS['holdout'], label='Development holdout'),
        patches.Patch(color=COLORS['evaluation'], label='Final June evaluation'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.88, top=0.95, bottom=0.08)
    fig.text(0.13, 0.015,
             'Training counts reflect outcome availability; June outcomes are assessed at approval + 45 days.',
             ha='left', va='bottom', fontsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    return fig
