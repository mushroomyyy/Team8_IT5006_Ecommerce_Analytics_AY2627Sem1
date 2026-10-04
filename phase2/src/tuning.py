"""Hyperparameter tuning, time-series cross-validation and threshold selection.

The default pipelines match `model_classification_dev.ipynb`. Every estimator uses
n_jobs=1 so parallelism happens only across CV folds/candidates, which avoids
oversubscribing CPU cores on laptops.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import RANDOM_STATE

CV_SCORING = {'roc_auc': 'roc_auc', 
              'pr_auc': 'average_precision', 
              'f1_at_0.5': 'f1'
              }

# Model family for each classifier (the brief allows 2-3 families in total)
MODEL_FAMILY = {
    'Logistic Regression': 'Linear',
    'Decision Tree': 'Tree-based',
    'Random Forest': 'Tree-based',
    'XGBoost': 'Tree-based',
    'LightGBM': 'Tree-based',
}

# Small random-search spaces; widen them once the feature set is frozen
CLASSIFIER_SEARCH_SPACES = {
    'Logistic Regression': {
        'classifier__C': np.logspace(-3, 4, 15),
    },
    'Decision Tree': {
        'classifier__max_depth': [4, 6, 8, 10, 12],
        'classifier__min_samples_leaf': [20, 35, 50, 75, 100],
        'classifier__criterion': ['gini', 'entropy'],
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [4, 6, 8, 10, 12],
        'classifier__min_samples_leaf': [1, 5, 10, 20, 50, 100],
        'classifier__max_features': ['sqrt', 0.3, 0.5],
    },
    'XGBoost': {
        'classifier__n_estimators': [200, 400, 600, 800],
        'classifier__max_depth': [2, 4, 6, 8],
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'classifier__subsample': [0.7, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__min_child_weight': [1, 5, 10, 20],
    },
    'LightGBM': {
        'classifier__n_estimators': [100, 200, 400],
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'classifier__num_leaves': [7, 15, 31, 63],
        'classifier__min_child_samples': [20, 35, 50, 75, 100],
        'classifier__subsample': [0.7, 1.0],
        'classifier__subsample_freq': [1],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__reg_lambda': [0.0, 1.0, 5.0],
    },
}


class FoldWeightedPipeline(Pipeline):
    """Recompute boosting class weight using only the labels passed to each fit.

    Inheriting Pipeline preserves cloning, classifier__ parameter search and
    named_steps access for feature importance and inference.
    """

    def fit(self, X, y=None, **params):
        labels = np.asarray(y)
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError('Boosting class weights require one-dimensional binary 0/1 labels.')
        negatives = np.count_nonzero(labels == 0)
        positives = np.count_nonzero(labels == 1)
        if not negatives or not positives:
            raise ValueError('Boosting training labels must contain both classes 0 and 1.')
        self.set_params(classifier__scale_pos_weight=float(negatives / positives))
        return super().fit(X, y, **params)


def build_classifiers(num_cols, cat_cols, random_state=RANDOM_STATE):
    """Return default pipelines, with boosting weights computed on every fit."""
    plain = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), num_cols),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], sparse_threshold=0)
    scaled = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), num_cols),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    return {
        'Logistic Regression': Pipeline(
            [('preprocessor', scaled), 
             ('classifier', LogisticRegression(
                class_weight='balanced', 
                max_iter=2000, 
                random_state=random_state
                ))
                ]
            ),
        'Decision Tree': Pipeline(
            [('preprocessor', clone(plain)), 
             ('classifier', DecisionTreeClassifier(
                max_depth=10, 
                class_weight='balanced', 
                random_state=random_state
                ))
                ]
            ),
        'Random Forest': Pipeline(
            [('preprocessor', clone(plain)), 
             ('classifier', RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                class_weight='balanced',
                random_state=random_state, 
                n_jobs=1
                ))
                ]
            ),
        'XGBoost': FoldWeightedPipeline(
            [('preprocessor', clone(plain)), 
             ('classifier', XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=random_state, 
                eval_metric='logloss', 
                n_jobs=1
                ))
                ]
            ),
        'LightGBM': FoldWeightedPipeline(
            [('preprocessor', clone(plain)), 
             ('classifier', LGBMClassifier(
                random_state=random_state,
                verbosity=-1, 
                n_jobs=1))
                ]
            ),
    }


def cv_summary(model, X, y, folds, scoring=CV_SCORING, n_jobs=-1):
    """Mean of each CV score across the given folds."""
    scores = cross_validate(model, X, y, cv=folds, scoring=scoring, n_jobs=n_jobs)
    row = {}
    for metric in scoring:
        values = scores[f'test_{metric}']
        row[f'cv_{metric}_mean'] = values.mean()
    return row


def tune_classifier(model, search_space, X, y, folds, n_iter, scoring='average_precision',
                    random_state=RANDOM_STATE, n_jobs=-1):
    """Random search over `search_space`; returns (unfitted tuned model, best params, search)."""
    search = RandomizedSearchCV(model, search_space, n_iter=n_iter, cv=folds, scoring=scoring,
                                random_state=random_state, n_jobs=n_jobs, refit=False)
    search.fit(X, y)
    return clone(model).set_params(**search.best_params_), search.best_params_, search


def out_of_fold_scores(model, X, y, folds):
    """Late-class scores for each validation row, fitted only on that fold's training rows.

    Rows that never fall in a validation fold (the earliest block) stay NaN.
    """
    scores = np.full(len(X), np.nan)
    for train_idx, valid_idx in folds:
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, list(fitted.classes_).index(1)]
    return scores


def best_f1_threshold(y_true, scores):
    """Threshold that maximises F1 on (y_true, scores); returns (threshold, f1)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    mask = ~np.isnan(scores)
    precision, recall, thresholds = precision_recall_curve(y_true[mask], scores[mask])
    f1 = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)
    best = int(np.nanargmax(f1[:-1]))  # The last point has no threshold
    return float(thresholds[best]), float(f1[best])


def params_to_json(params):
    """Convert numpy values in best_params_ to plain Python types for saving."""
    return {key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in params.items()}
