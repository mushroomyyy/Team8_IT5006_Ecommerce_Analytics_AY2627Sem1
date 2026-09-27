"""Metrics and coverage tables for the classification and regression tracks."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, median_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None):
    """Imbalance-aware classification metrics; the late class (1) is positive."""
    two_classes = pd.Series(y_true).nunique() == 2
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None and two_classes else np.nan,
        'pr_auc': average_precision_score(y_true, y_prob) if y_prob is not None and two_classes else np.nan,
    }


def evaluate_pipeline(pipeline, X_test, y_test, threshold=0.5):
    """Return (metrics, y_pred, y_prob) for a fitted classification pipeline on raw features."""
    y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    y_pred = (y_prob >= threshold).astype(int)
    return classification_metrics(y_test, y_pred, y_prob), y_pred, y_prob


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R², plus median absolute error and mean bias (pred - actual)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'bias': float(np.mean(y_pred - y_true)),
    }


def get_coverage(prob, y_true):
    """Precision and recall (coverage) by predicted-risk decile; decile 1 is riskiest."""
    predictions_df = pd.DataFrame({'actual_late_delivery': y_true,
                                   'prob_late_delivery': prob}, index=y_true.index)
    bins = pd.qcut(predictions_df['prob_late_delivery'], q=10, labels=False, duplicates='drop')
    # Identical scores form one group rather than silently dropping all rows
    predictions_df['decile'] = (pd.Series(1, index=predictions_df.index, dtype='Int64')
                                if bins.isna().all() else (bins.max() - bins + 1).astype('Int64'))
    summary = predictions_df.groupby('decile').agg(
        number_of_orders=('actual_late_delivery', 'size'),
        positive_labels=('actual_late_delivery', 'sum'),
    ).sort_index()
    summary['cumulative_orders'] = summary['number_of_orders'].cumsum()
    summary['cumulative_positive_labels'] = summary['positive_labels'].cumsum()
    summary['late_delivery_rate'] = summary['positive_labels'] / summary['number_of_orders']
    summary['cumulative_late_delivery_rate'] = (summary['cumulative_positive_labels']
                                                / summary['cumulative_orders'])
    total_positive = summary['positive_labels'].sum()
    summary['coverage'] = (summary['cumulative_positive_labels'] / total_positive
                           if total_positive > 0 else float('nan'))
    return summary.reset_index()
