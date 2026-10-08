"""Class-imbalance experiment: resampling applied only inside each training fold.

Late orders are about 8% of Train. This module wraps a model pipeline with a
random under-sampler, a random over-sampler or SMOTENC (SMOTE for mixed numeric and
categorical data). It uses `imblearn.pipeline.Pipeline`, which calls the sampler
during `fit` only, so validation folds, Validation, June and later months are
never resampled. Numeric columns are median-imputed before the sampler because
SMOTENC cannot handle missing values (the model's own imputer then sees complete data).
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from . import RANDOM_STATE
from .evaluation import CLASSIFICATION_CV_METRICS, classification_eval_metrics, cv_classification_metrics

STRATEGIES = {'none': 'No resampling', 'undersample': 'Random undersampling',
              'oversample': 'Random oversampling', 'smotenc': 'SMOTENC'}


def _is_binary(column):
    return set(column.dropna().unique()) <= {0, 1}


def binary_columns(X, num_cols):
    """Numeric columns that only hold 0/1 flags; SMOTENC treats them as categorical."""
    return [c for c in num_cols if _is_binary(X[c])]


def make_sampler(strategy, categorical_positions, random_state=RANDOM_STATE):
    """Sampler that balances the classes 1:1 (the default `sampling_strategy='auto'` for SMOTENC)."""
    if strategy == 'undersample':
        return RandomUnderSampler(random_state=random_state)
    if strategy == 'oversample':
        return RandomOverSampler(random_state=random_state)
    if strategy == 'smotenc':
        return SMOTENC(categorical_features=categorical_positions, random_state=random_state)
    raise ValueError(f'Unknown resampling strategy: {strategy}')


def with_resampling(model, strategy, num_cols, cat_cols, flag_cols=(), random_state=RANDOM_STATE):
    """Return `model` unchanged ('none') or an imblearn pipeline that resamples before it.

    `model` is a pipeline whose steps work on the raw feature frame. The first step
    imputes the numeric columns (the model imputes identically, so nothing else changes);
    the sampler follows, then the original model pipeline. The flags and categorical columns are
    the SMOTENC categorical features.
    """
    if strategy == 'none':
        return clone(model)
    columns = list(num_cols) + list(cat_cols)
    impute = ColumnTransformer(
        [('numeric', SimpleImputer(strategy='median'), list(num_cols)),
         ('categorical', 'passthrough', list(cat_cols))],
        verbose_feature_names_out=False).set_output(transform='pandas')
    positions = [columns.index(c) for c in list(flag_cols) + list(cat_cols)]
    steps = [('impute', impute), ('sampler', make_sampler(strategy, positions, random_state)),
             ('model', clone(model))]
    return ImbPipeline(steps)


def paired_gain(cv_table, model, strategy, metric='avg_precision'):
    """Mean and SE of the per-fold change in `metric` versus no resampling."""
    pick = lambda s: cv_table[(cv_table['model'] == model) & (cv_table['strategy'] == s)].sort_values('fold')
    diff = pick(strategy)[metric].to_numpy() - pick('none')[metric].to_numpy()
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def run_resampling_experiment(models, X_train, y_train, folds, X_val, y_val, num_cols, cat_cols,
                              flag_cols, strategies=tuple(STRATEGIES)):
    """CV (same chronological folds) and Validation scores for each model and strategy.

    `models` maps a label (e.g. 'C1') to an unfitted pipeline. Returns
    (cv_folds, summary, validation_probabilities): per-fold scores, one row per
    model and strategy (CV mean and SD, Validation scores, paired AP gain), and the
    Validation probabilities for calibration plots.
    """
    folds_out, validation, probabilities = [], [], {}
    for label, model in models.items():
        for strategy in strategies:
            pipeline = with_resampling(model, strategy, num_cols, cat_cols, flag_cols)
            fold_scores = cv_classification_metrics(pipeline, X_train, y_train, folds)
            folds_out.append(fold_scores.assign(model=label, strategy=strategy))
            fitted = clone(pipeline).fit(X_train, y_train)
            prob = fitted.predict_proba(X_val)[:, list(fitted.classes_).index(1)]
            probabilities[(label, strategy)] = prob
            scores = classification_eval_metrics(y_val, prob)
            validation.append({'model': label, 'strategy': strategy,
                               **{f'val_{m}': scores[m] for m in CLASSIFICATION_CV_METRICS},
                               'val_mean_probability': float(prob.mean()),
                               'val_late_rate': float(np.mean(y_val))})
    cv_table = pd.concat(folds_out, ignore_index=True)
    summary = cv_table.drop(columns='fold').groupby(['model', 'strategy'], sort=False).agg(['mean', 'std'])
    summary.columns = [f'cv_{m}_{s}' for m, s in summary.columns]
    summary = summary.reset_index().merge(pd.DataFrame(validation), on=['model', 'strategy'])
    gains = [paired_gain(cv_table, r.model, r.strategy) if r.strategy != 'none' else (0.0, 0.0)
             for r in summary.itertuples()]
    summary['ap_gain_vs_none'] = [g[0] for g in gains]
    summary['ap_gain_se'] = [g[1] for g in gains]
    summary['clear_ap_gain'] = summary['ap_gain_vs_none'] > 2 * summary['ap_gain_se']
    summary.loc[summary['strategy'] == 'none', 'clear_ap_gain'] = False
    return cv_table, summary, probabilities
