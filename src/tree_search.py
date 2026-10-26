"""Single decision tree (C4 / R4): a CV grid search on Train, with cached outputs.

The tree is the tree-family baseline for the Random Forest. It uses the same raw-feature
preprocessing, Train rows and chronological CV folds as `src.rf_search`. Ties within
`tie_tol` of the best mean CV score go to the tree with the fewest leaves.
Run with ``python -m src.tree_search --task classification|regression``.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import RANDOM_STATE
from .preprocessing import make_preprocessor
from .rf_search import OUTPUT_ROOT, SCORING, STEP, _plain

DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 16, None]
LEAVES = [1, 5, 10, 20, 50, 100, 200, 500]


def search_space(task):
    space = {'max_depth': DEPTHS, 'min_samples_leaf': LEAVES}
    if task == 'classification':
        space['criterion'] = ['gini', 'entropy']
    return space


def build_tree_pipeline(task, num_cols, cat_cols, random_state=RANDOM_STATE, **params):
    """Raw-feature decision tree pipeline; `params` set the tree (no step prefix)."""
    tree = DecisionTreeClassifier if task == 'classification' else DecisionTreeRegressor
    return Pipeline([('preprocessor', make_preprocessor(num_cols, cat_cols, reference_categories=True)),
                     (STEP[task], tree(random_state=random_state, **params))])


def run_tree_search(task, train_df, y, folds, num_cols, cat_cols, output_dir=None, tie_tol=1e-4, n_jobs=-1):
    """Grid search on Train CV; writes `tree_search.csv` and `tree_best_params.json`."""
    output_dir = Path(output_dir or OUTPUT_ROOT / task)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = train_df[num_cols + cat_cols].reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    prefix = f'{STEP[task]}__'
    started = time.time()
    search = GridSearchCV(build_tree_pipeline(task, num_cols, cat_cols),
                          {prefix + k: v for k, v in search_space(task).items()},
                          cv=folds, scoring=SCORING[task], n_jobs=n_jobs, refit=False)
    search.fit(X, y)
    frame = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(list(frame['params'])).rename(columns=lambda c: c[len(prefix):])
    results = pd.concat([params, pd.DataFrame({'mean_score': frame['mean_test_score'],
                                               'sd_score': frame['std_test_score']})], axis=1)
    # Leaves of each candidate refitted on all Train rows, for the tie-break
    results['n_leaves'] = [
        clone(build_tree_pipeline(task, num_cols, cat_cols, **{k: _plain(v, k) for k, v in row.items()}))
        .fit(X, y).steps[-1][1].get_n_leaves()
        for row in params.to_dict('records')]
    results = results.sort_values('mean_score', ascending=False, ignore_index=True)
    near = results[results['mean_score'] >= results['mean_score'].max() - tie_tol]
    best_row = near.sort_values(['n_leaves', 'mean_score'], ascending=[True, False]).iloc[0]
    best_params = {k: _plain(best_row[k], k) for k in search_space(task)}
    metadata = {'task': task, 'scoring': SCORING[task], 'candidates': len(results), 'tie_tolerance': tie_tol,
                'train_rows': len(X), 'n_cv_folds': len(folds), 'best_cv_mean': float(best_row['mean_score']),
                'best_cv_sd': float(best_row['sd_score']), 'best_n_leaves': int(best_row['n_leaves']),
                'seconds': round(time.time() - started, 1), 'random_state': RANDOM_STATE,
                'run_date': pd.Timestamp.now().isoformat(timespec='seconds')}
    results.to_csv(output_dir / 'tree_search.csv', index=False)
    result = {'best_params': best_params, 'metadata': metadata}
    (output_dir / 'tree_best_params.json').write_text(json.dumps(result, indent=2))
    return {'search': results, **result}


def load_or_run_tree_search(task, run=False, data=None, output_dir=None, **kwargs):
    """Load the cached tree search, or run it when `run=True` (needs `data` from `build_task_data`)."""
    output_dir = Path(output_dir or OUTPUT_ROOT / task)
    if run:
        if data is None:
            raise ValueError('Pass data=build_task_data(...) to run the search')
        return run_tree_search(task, data.train_df, data.y_train, data.folds, data.num_cols,
                               data.cat_cols, output_dir=output_dir, **kwargs)
    cached = output_dir / 'tree_best_params.json'
    if not cached.exists():
        raise FileNotFoundError(f'No cached tree search at {cached}; run '
                                f'`python -m src.tree_search --task {task}`')
    result = json.loads(cached.read_text())
    result['search'] = pd.read_csv(output_dir / 'tree_search.csv')
    return result


def tuned_tree_pipeline(task, best_params, num_cols, cat_cols):
    """Unfitted decision tree pipeline with the tuned parameters."""
    return build_tree_pipeline(task, num_cols, cat_cols, **{k: _plain(v, k) for k, v in best_params.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--task', choices=['classification', 'regression'], required=True)
    args = parser.parse_args(argv)
    from .data import load_olist_data
    from .datasets import build_task_data
    from .features import add_extra_features, build_feature_table

    olist = load_olist_data(verbose=False)
    data = build_task_data(args.task, add_extra_features(build_feature_table(olist), olist))
    result = run_tree_search(args.task, data.train_df, data.y_train, data.folds, data.num_cols, data.cat_cols)
    print(json.dumps(result['best_params']), json.dumps(result['metadata']))


if __name__ == '__main__':
    main()
