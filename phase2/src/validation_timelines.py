"""Timeline figure: Train, CV, Validation, Test (June) and Monitoring (July, August).

`task_timeline` collects the dates and order counts of one task from its `TaskData`;
`plot_validation_timelines` draws the classification and regression panels together.
"""
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from .labels import attach_prediction_labels, attach_regression_actuals
from .report_figures import AQUA, BLUE, INK, INK2, ORANGE, PURPLE, SLATE, STYLE, UNUSED

COLORS = {'train': BLUE, 'buffer': UNUSED, 'cv': ORANGE, 'validation': AQUA,
          'test': PURPLE, 'monitoring': SLATE}
MONTHS = [('Test (June)', '2018-06-01', 'test'), ('Monitoring (July)', '2018-07-01', 'monitoring'),
          ('Monitoring (August)', '2018-08-01', 'monitoring')]


def cohort_counts(final_df, orders, task, month_start, lookback):
    """Orders approved in a month that are scored, and those whose outcome is known at +45 days.

    Classification scores orders that were not already delivered on their approval day.
    """
    start = pd.Timestamp(month_start)
    end = start + pd.offsets.MonthEnd(0)
    cohort = final_df[final_df['order_approved_dt'].between(start, end)]
    if task == 'classification':
        delivered = pd.to_datetime(cohort['order_delivered_customer_date']).dt.normalize()
        cohort = cohort[delivered.isna() | delivered.gt(cohort['order_approved_dt'])]
        stub = cohort[['order_id', 'order_approved_dt']].assign(predicted_probability=0.0)
        evaluated = attach_prediction_labels(stub, orders, lookback=lookback)['actual_label'].notna().sum()
    else:
        stub = cohort[['order_id', 'order_approved_dt']]
        evaluated = attach_regression_actuals(stub, orders, waiting_period=lookback)['actual_target'].notna().sum()
    return {'start': start, 'end': end, 'scored': len(cohort), 'evaluated': int(evaluated)}


def task_timeline(task_data, final_df, orders, lookback):
    """Everything the timeline panel needs for one task."""
    return {'name': task_data.task.capitalize(), 'train': task_data.train_df,
            'validation': task_data.validation_df, 'history': task_data.history_df,
            'folds': task_data.folds, 'window': task_data.window,
            'cohorts': [{'label': label, 'role': role,
                         **cohort_counts(final_df, orders, task_data.task, start, lookback)}
                        for label, start, role in MONTHS]}


def _bar(ax, start, end, y, color, text=None, text_color=INK, height=0.58):
    """Draw an inclusive calendar-date interval as a horizontal bar."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    width = max((end - start).days + 1, 1)
    ax.barh(y, width, left=mdates.date2num(start), height=height, color=color,
            edgecolor='white', linewidth=0.6)
    if text:
        ax.text(mdates.date2num(start) + width / 2, y, text, ha='center', va='center',
                fontsize=8, color=text_color, clip_on=True)


def _note(ax, start, end, y, text, ha='center'):
    """Caption above a bar, centred on the interval."""
    x = mdates.date2num(start) + (mdates.date2num(end) - mdates.date2num(start)) / 2
    ax.text(x, y + 0.34, text, ha=ha, va='bottom', fontsize=7.4, color=INK2, clip_on=False)


def _days(rows):
    dates = pd.to_datetime(rows['order_approved_dt']).dt.normalize()
    return dates.min(), dates.max()


def _draw_panel(ax, timeline, gap_days):
    train, validation, folds = timeline['train'], timeline['validation'], timeline['folds']
    window_start, window_end = (pd.Timestamp(d) for d in timeline['window'])
    labels = [f'CV fold {i}' for i in range(1, len(folds) + 1)] + [
        'Train and\nValidation', 'Refit and\nTest (June)', 'Monitoring\n(frozen models)']
    ys = list(range(len(labels) - 1, -1, -1))
    ax.set_yticks(ys, labels)
    ax.set_title(timeline['name'], loc='left', fontsize=13, fontweight='bold', pad=11)
    train_dates = pd.to_datetime(train['order_approved_dt']).dt.normalize()
    first_valid = train_dates.max() - pd.Timedelta(days=len(folds) * 30 - 1)

    for i, (train_idx, valid_idx) in enumerate(folds):
        start, end = _days(train.iloc[train_idx])
        valid_start = first_valid + pd.Timedelta(days=30 * i)
        valid_end = valid_start + pd.Timedelta(days=29)
        _bar(ax, start, end, ys[i], COLORS['train'], f'{len(train_idx):,} Train', 'white')
        _bar(ax, valid_start - pd.Timedelta(days=gap_days), valid_start - pd.Timedelta(days=1),
             ys[i], COLORS['buffer'], f'{gap_days}d')
        _bar(ax, valid_start, valid_end, ys[i], COLORS['cv'])
        _note(ax, valid_start, valid_end, ys[i],
              f'{valid_start:%d %b}-{valid_end:%d %b} | {len(valid_idx):,} orders')

    y = ys[len(folds)]
    start, end = _days(train)
    val_start, val_end = _days(validation)
    _bar(ax, start, end, y, COLORS['train'], f'{len(train):,} Train', 'white')
    _bar(ax, val_start - pd.Timedelta(days=gap_days), val_start - pd.Timedelta(days=1), y,
         COLORS['buffer'], f'{gap_days}d')
    _bar(ax, val_start, val_end, y, COLORS['validation'])
    _note(ax, val_start, val_end, y, f'Validation {val_start:%d %b}-{val_end:%d %b} | {len(validation):,} orders')

    y = ys[len(folds) + 1]
    test = timeline['cohorts'][0]
    _bar(ax, window_start, window_end, y, COLORS['train'],
         f'Refit on all known outcomes | {len(timeline["history"]):,} orders', 'white')
    _bar(ax, window_end + pd.Timedelta(days=1), test['start'] - pd.Timedelta(days=1), y,
         COLORS['buffer'], f'{gap_days}d cut-off')
    _bar(ax, test['start'], test['end'], y, COLORS['test'])
    _note(ax, test['start'], test['end'], y,
          f'Test (June) | {test["scored"]:,} scored, {test["evaluated"]:,} evaluated', ha='right')

    y = ys[-1]
    monitoring = timeline['cohorts'][1:]
    for cohort in monitoring:
        _bar(ax, cohort['start'], cohort['end'], y, COLORS['monitoring'], cohort['label'].split('(')[1][:-1],
             'white')
    ax.text(mdates.date2num(monitoring[0]['start']) - 3, y,
            ' | '.join(f'{c["label"].split("(")[1][:-1]}: {c["scored"]:,} scored, {c["evaluated"]:,} evaluated'
                       for c in monitoring), ha='right', va='center', fontsize=7.4, color=INK2)

    ax.grid(axis='x', color='#e8e8e8', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.tick_params(axis='x', labelsize=8, labelbottom=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(window_start - pd.Timedelta(days=8), timeline['cohorts'][-1]['end'] + pd.Timedelta(days=10))
    ax.set_xlabel('Order approval date')


def plot_validation_timelines(timelines, output_path, gap_days=45):
    """Save the stacked task panels (one `task_timeline` mapping per panel) as a PNG."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(timelines), 1, figsize=(16, 5.0 * len(timelines)),
                                 gridspec_kw={'hspace': 0.5})
        for ax, timeline in zip(axes, timelines):
            _draw_panel(ax, timeline, gap_days)
        legend = [patches.Patch(color=COLORS['train'], label='Train / refit'),
                  patches.Patch(color=COLORS['buffer'], label='Excluded gap'),
                  patches.Patch(color=COLORS['cv'], label='CV validation'),
                  patches.Patch(color=COLORS['validation'], label='Validation'),
                  patches.Patch(color=COLORS['test'], label='Test (June)'),
                  patches.Patch(color=COLORS['monitoring'], label='Monitoring (July, August)')]
        fig.legend(handles=legend, loc='upper center', ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, 0.94), fontsize=9)
        fig.suptitle('Train, CV, Validation, Test (June) and Monitoring timeline', x=0.02, y=0.99,
                     ha='left', fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.11)
        fig.text(0.1, 0.005,
                 'All decisions use Train only. Validation is a diagnostic. Test (June) is scored daily by models refit '
                 'on all outcomes known before 2 June.\nMonitoring scores the same frozen models on July and August '
                 'with no refit. "Evaluated" counts orders whose outcome is known 45 days after approval.',
                 ha='left', va='bottom', fontsize=8.5, color=INK2)
        fig.savefig(output_path, dpi=200)
    return fig
