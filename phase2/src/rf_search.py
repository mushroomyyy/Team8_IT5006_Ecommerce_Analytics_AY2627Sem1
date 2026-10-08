"""Wide two-stage Random Forest search on Train, with cached outputs.

Run with ``python -m src.rf_search --task classification|regression`` from ``phase2``.
Stage A is a broad randomised search (n_estimators fixed at 200, refit=False). Stage B is
a small grid around the Stage A best with n_estimators in {300, 500}; ties go to the
simpler forest. The search uses Train rows and the chronological CV folds only.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import RANDOM_STATE
from .preprocessing import make_preprocessor

OUTPUT_ROOT = Path('results/simple_to_complex')
STEP = {'classification': 'classifier', 'regression': 'model'}
SCORING = {'classification': 'average_precision', 'regression': 'neg_root_mean_squared_error'}

# Ordered from simplest to most flexible; Stage B uses the order to find neighbours.
DEPTHS = [6, 8, 10, 12, 16, 20, 30, None]
LEAVES = [1, 2, 5, 10, 20, 50, 100, 200]
SPLITS = [2, 5, 10, 20, 50]
FEATURES = ['log2', 'sqrt', 0.2, 0.3, 0.5, 0.7, 1.0]
SAMPLES = [None, 0.5, 0.7, 0.9]


def search_space(task):
    """Stage A space (parameter names without the pipeline step prefix)."""
    space = {'max_depth': [None] + DEPTHS[:-1], 'min_samples_leaf': LEAVES,
             'min_samples_split': SPLITS, 'max_features': FEATURES, 'max_samples': SAMPLES}
    if task == 'classification':
        space['class_weight'] = [None, 'balanced', 'balanced_subsample']
        space['criterion'] = ['gini', 'entropy']
    return space


def build_rf_pipeline(task, num_cols, cat_cols, n_estimators=200, random_state=RANDOM_STATE,
                      **params):
    """Raw-feature Random Forest pipeline; `params` set the forest (no step prefix)."""
    if task == 'classification':
        forest = RandomForestClassifier
    else:
        forest = RandomForestRegressor
    extra = {'criterion': 'squared_error'} if task == 'regression' else {}
    model = forest(n_estimators=n_estimators, bootstrap=True, random_state=random_state,
                   n_jobs=1, **{**extra, **params})
    return Pipeline([('preprocessor', make_preprocessor(num_cols, cat_cols, reference_categories=True)),
                     (STEP[task], model)])


def _prefixed(task, params):
    return {f'{STEP[task]}__{key}': value for key, value in params.items()}


def _neighbours(values, best, width):
    index = values.index(best)
    return values[max(0, index - width): index + width + 1]


def stage_b_grid(task, best, n_estimators=(300, 500), width=1):
    """Neighbouring max_depth, min_samples_leaf and max_features values around `best`."""
    grid = {'max_depth': _neighbours(DEPTHS, best['max_depth'], width),
            'min_samples_leaf': _neighbours(LEAVES, best['min_samples_leaf'], width),
            'max_features': _neighbours(FEATURES, best['max_features'], width),
            'n_estimators': list(n_estimators)}
    fixed = {k: [v] for k, v in best.items() if k not in grid and k != 'n_estimators'}
    return {**fixed, **grid}


def _complexity(row):
    """Sort key for ties: shallower, larger leaves, fewer trees."""
    depth = np.inf if row['max_depth'] is None or pd.isna(row['max_depth']) else row['max_depth']
    return (depth, -row['min_samples_leaf'], row.get('n_estimators', 0))


def _results_frame(search_results, prefix):
    frame = pd.DataFrame(search_results)
    params = pd.DataFrame(list(frame['params'])).rename(columns=lambda c: c[len(prefix):])
    folds = [c for c in frame.columns if c.startswith('split') and c.endswith('_test_score')]
    out = pd.concat([params, pd.DataFrame({
        'mean_score': frame['mean_test_score'], 'sd_score': frame['std_test_score'],
        'mean_fit_time': frame['mean_fit_time'], 'mean_score_time': frame['mean_score_time'],
    }), frame[folds]], axis=1)
    return out.sort_values('mean_score', ascending=False, ignore_index=True)


def _pick_best(results, tie_tol):
    """Best mean score; within `tie_tol` of it, the simplest forest."""
    near = results[results['mean_score'] >= results['mean_score'].max() - tie_tol]
    order = sorted(near.index, key=lambda i: _complexity(near.loc[i]))
    return near.loc[order[0]]


INT_KEYS = {'max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators'}


def _plain(value, key=None):
    """Plain Python value: numpy scalars unwrapped, NaN as None, integer parameters as int."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if key in INT_KEYS and value is not None:
        return int(value)
    return value


def run_rf_search(task, train_df, y, folds, num_cols, cat_cols, output_dir=None, figure_dir=None,
                  n_iter=60, stage_a_trees=200, stage_b_trees=(300, 500), stage_b_width=1,
                  tie_tol=1e-4, n_jobs=-1, extra_metadata=None):
    """Run both stages on Train and write the CSVs, JSON and sensitivity plot.

    `train_df` must be the Train rows only (CV folds index into it positionally).
    Returns {'stage_a', 'stage_b', 'best_params', 'metadata'}.
    """
    output_dir = Path(output_dir or OUTPUT_ROOT / task)
    figure_dir = Path(figure_dir or OUTPUT_ROOT / 'figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    scoring, prefix = SCORING[task], f'{STEP[task]}__'
    X = train_df[num_cols + cat_cols]
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    started = time.time()

    base = build_rf_pipeline(task, num_cols, cat_cols, n_estimators=stage_a_trees)
    search = RandomizedSearchCV(base, _prefixed(task, search_space(task)), n_iter=n_iter, cv=folds,
                                scoring=scoring, random_state=RANDOM_STATE, n_jobs=n_jobs,
                                refit=False)
    search.fit(X, y)
    stage_a = _results_frame(search.cv_results_, prefix)
    stage_a_seconds = time.time() - started
    best_a = {k: _plain(v, k) for k, v in _pick_best(stage_a, tie_tol).items()
              if k in search_space(task)}

    started_b = time.time()
    grid = stage_b_grid(task, best_a, stage_b_trees, stage_b_width)
    search_b = GridSearchCV(base, _prefixed(task, grid), cv=folds, scoring=scoring,
                            n_jobs=n_jobs, refit=False)
    search_b.fit(X, y)
    stage_b = _results_frame(search_b.cv_results_, prefix)
    stage_b_seconds = time.time() - started_b
    best_row = _pick_best(stage_b, tie_tol)
    best_params = {k: _plain(best_row[k], k) for k in list(grid)}
    # Fixed-at-Stage-A values travel with the best parameters
    best_params = {**best_params, **{k: _plain(best_row[k], k) for k in best_a if k not in best_params}}

    metadata = {
        'task': task, 'scoring': scoring, 'n_iter_stage_a': n_iter,
        'stage_a_trees': stage_a_trees, 'stage_b_trees': list(stage_b_trees),
        'stage_b_width': stage_b_width, 'stage_b_candidates': len(stage_b),
        'tie_tolerance': tie_tol, 'train_rows': len(X), 'n_cv_folds': len(folds),
        'features': {'numeric': num_cols, 'categorical': cat_cols},
        'best_cv_mean': float(best_row['mean_score']), 'best_cv_sd': float(best_row['sd_score']),
        'stage_a_best_cv_mean': float(stage_a['mean_score'].max()),
        'stage_a_seconds': round(stage_a_seconds, 1), 'stage_b_seconds': round(stage_b_seconds, 1),
        'run_date': pd.Timestamp.now().isoformat(timespec='seconds'),
        'random_state': RANDOM_STATE,
        'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        **(extra_metadata or {}),
    }
    stage_a.to_csv(output_dir / 'rf_search_stageA.csv', index=False)
    stage_b.to_csv(output_dir / 'rf_search_stageB.csv', index=False)
    result = {'best_params': best_params, 'stage_a_best_params': best_a, 'metadata': metadata}
    (output_dir / 'rf_best_params.json').write_text(json.dumps(result, indent=2))
    plot_rf_sensitivity(stage_a, task, figure_dir / f'rf_sensitivity_{task}.png')
    return {'stage_a': stage_a, 'stage_b': stage_b, **result}


def plot_rf_sensitivity(stage_a, task, save_path):
    """CV score vs max_depth and vs min_samples_leaf from the Stage A candidates."""
    import matplotlib.pyplot as plt
    from . import report_figures as figs

    label = 'CV average precision' if task == 'classification' else 'CV RMSE (days)'
    sign = 1 if task == 'classification' else -1  # Plot RMSE in natural units
    frame = stage_a.assign(score=sign * stage_a['mean_score'],
                           depth=stage_a['max_depth'].fillna(40).astype(float))
    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, column, title in [(axes[0], 'depth', 'max_depth (None shown at 40)'),
                                  (axes[1], 'min_samples_leaf', 'min_samples_leaf')]:
            ax.scatter(frame[column], frame['score'], s=18, alpha=0.5, color=figs.BLUE)
            best = frame.groupby(column)['score'].max()
            ax.plot(best.index, best.values, color=figs.ORANGE, marker='o', label='best per value')
            ax.set(xlabel=title, ylabel=label)
            if column == 'min_samples_leaf':
                ax.set_xscale('log')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)


def load_or_run_rf_search(task, run=False, data=None, output_dir=None, **kwargs):
    """Load cached search outputs, or run the search when `run=True`.

    `data` is the `TaskData` from `build_task_data` (needed only to run). Returns the
    same dict as `run_rf_search`.
    """
    output_dir = Path(output_dir or OUTPUT_ROOT / task)
    cached = output_dir / 'rf_best_params.json'
    if run:
        if data is None:
            raise ValueError('Pass data=build_task_data(...) to run the search')
        return run_rf_search(task, data.train_df, data.y_train, data.folds, data.num_cols,
                             data.cat_cols, output_dir=output_dir, **kwargs)
    if not cached.exists():
        raise FileNotFoundError(f'No cached RF search at {cached}; run '
                                f'`python -m src.rf_search --task {task}`')
    result = json.loads(cached.read_text())
    result['stage_a'] = pd.read_csv(output_dir / 'rf_search_stageA.csv')
    result['stage_b'] = pd.read_csv(output_dir / 'rf_search_stageB.csv')
    return result


def tuned_rf_pipeline(task, best_params, num_cols, cat_cols, n_jobs=-1):
    """Unfitted Random Forest pipeline with the tuned parameters (n_estimators included)."""
    params = {k: _plain(v, k) for k, v in best_params.items()}
    n_estimators = params.pop('n_estimators')
    pipeline = build_rf_pipeline(task, num_cols, cat_cols, n_estimators=n_estimators, **params)
    return pipeline.set_params(**{f'{STEP[task]}__n_jobs': n_jobs})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--task', choices=['classification', 'regression'], required=True)
    parser.add_argument('--n-iter', type=int, default=60, help='Stage A candidates (60; 40 if slow)')
    parser.add_argument('--stage-a-trees', type=int, default=200)
    parser.add_argument('--stage-b-trees', type=int, nargs='+', default=[300, 500])
    parser.add_argument('--stage-b-width', type=int, default=1,
                        help='Neighbours on each side of the Stage A best (0 = refit only n_estimators)')
    parser.add_argument('--output-dir', help='Defaults to results/simple_to_complex/<task>')
    parser.add_argument('--figure-dir', help='Defaults to results/simple_to_complex/figures')
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    from .data import load_olist_data
    from .datasets import build_task_data
    from .features import add_extra_features, build_feature_table

    olist = load_olist_data(verbose=False)
    data = build_task_data(args.task, add_extra_features(build_feature_table(olist), olist))
    print(f'{args.task}: Train {len(data.train_df):,} rows, {len(data.folds)} folds', flush=True)
    result = run_rf_search(
        args.task, data.train_df, data.y_train, data.folds, data.num_cols, data.cat_cols,
        output_dir=args.output_dir, figure_dir=args.figure_dir, n_iter=args.n_iter,
        stage_a_trees=args.stage_a_trees, stage_b_trees=tuple(args.stage_b_trees),
        stage_b_width=args.stage_b_width, extra_metadata={'row_counts': data.counts()})
    meta = result['metadata']
    print(f"Stage A {meta['stage_a_seconds']}s, Stage B {meta['stage_b_seconds']}s; "
          f"best CV {meta['best_cv_mean']:.4f} +/- {meta['best_cv_sd']:.4f}")
    print(json.dumps(result['best_params']))


if __name__ == '__main__':
    main()
