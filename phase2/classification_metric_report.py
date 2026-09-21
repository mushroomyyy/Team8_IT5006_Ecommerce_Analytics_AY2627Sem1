"""Report complementary metrics from saved OOF probabilities/forward decisions.

No fitting or threshold selection. AP remains the primary ranking measure.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, auc,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
    brier_score_loss, log_loss)
from classification_experiments import ROOT
from model_workbench import coverage_metrics


def summarise(pred):
    y, probability, decision = pred.actual, pred.probability.to_numpy(), pred.predicted
    if not decision.isin([0, 1]).all():
        raise ValueError('Forward decisions must be saved binary values')
    fold_rows = []
    for fold, q in pred.groupby('fold'):
        precision, recall, _ = precision_recall_curve(q.actual, q.probability)
        fold_rows.append(dict(fold=int(fold), late_rate=float(q.actual.mean()),
            average_precision=float(average_precision_score(q.actual, q.probability)),
            pr_auc_trapezoidal=float(auc(recall, precision)),
            roc_auc=float(roc_auc_score(q.actual, q.probability))))
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    capacity = coverage_metrics(y, probability, dates=pred.approval_date)
    for fraction in [5, 10, 20, 30]:
        captured = capacity[f'coverage_at_{fraction}pct'] * y.sum()
        selected = capacity[f'selected_fraction_at_{fraction}pct'] * len(y)
        capacity[f'precision_at_{fraction}pct'] = float(captured / selected)
    brier = float(brier_score_loss(y, probability))
    return dict(rows=len(pred), late_rate=float(y.mean()),
        mean_fold_ap=float(np.mean([q['average_precision'] for q in fold_rows])),
        mean_fold_pr_auc_trapezoidal=float(np.mean([q['pr_auc_trapezoidal'] for q in fold_rows])),
        mean_fold_roc_auc=float(np.mean([q['roc_auc'] for q in fold_rows])),
        forward_precision=float(precision_score(y, decision, zero_division=0)),
        forward_recall=float(recall_score(y, decision, zero_division=0)),
        forward_f1=float(f1_score(y, decision, zero_division=0)),
        forward_accuracy=float(accuracy_score(y, decision)),
        forward_balanced_accuracy=float(balanced_accuracy_score(y, decision)),
        forward_mcc=float(matthews_corrcoef(y, decision)),
        forward_specificity=float(tn / (tn + fp)),
        forward_alert_fraction=float(decision.mean()),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        brier_score=brier, probability_rmse=float(np.sqrt(brier)),
        log_loss=float(log_loss(y, probability, labels=[0, 1])),
        capacity=capacity, fold_ranking=fold_rows)


def run():
    frame = pd.read_csv(ROOT / 'classification_ensemble_predictions.csv', parse_dates=['approval_date'])
    dummy = pd.read_csv(ROOT / 'classification_backtest_predictions.csv', parse_dates=['approval_date'])
    dummy = dummy.loc[dummy.model.eq('dummy_prior') & dummy['sample'].eq('cross_validation')]
    frame = pd.concat([frame, dummy], ignore_index=True)
    reference = frame.loc[frame.model.eq('base_logistic_regression')].sort_values('order_id')
    results = {}
    for name, pred in frame.groupby('model'):
        pred = pred.sort_values('order_id')
        assert pred.order_id.is_unique
        for col in ['order_id', 'actual', 'approval_date', 'fold']:
            assert np.array_equal(reference[col].to_numpy(), pred[col].to_numpy())
        assert pred.probability.between(0, 1).all()
        results[name] = summarise(pred)
    artifact = dict(status='development_metrics_no_refitting', primary='equal-month mean average precision',
        decision_policy='Saved prospective thresholds from earlier mature labels; not refitted for this report.',
        models=results)
    (ROOT / 'classification_metric_summary.json').write_text(json.dumps(artifact, indent=2, allow_nan=False))
    lines = ['# Classification metric comparison', '',
        'Generated from paired development predictions; no refitting or threshold optimisation.', '',
        'Ranking metrics are equal-month means. Decision/probability metrics pool orders. Capacity selection is within each day.', '',
        '## Ranking and probability quality', '',
        '| Model | AP ↑ | Trapezoidal PR-AUC ↑ | ROC-AUC ↑ | Brier/MSE ↓ | Log loss ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {name} | {r['mean_fold_ap']:.4f} | {r['mean_fold_pr_auc_trapezoidal']:.4f} | {r['mean_fold_roc_auc']:.4f} | {r['brier_score']:.4f} | {r['log_loss']:.4f} |")
    lines += ['', 'The constant-per-fold dummy illustrates why AP and trapezoidal PR-AUC must not be conflated: trapezoidal endpoint interpolation can give a misleadingly large area for constant scores. AP is the primary measure.', '',
        '## Decisions under forward thresholds', '',
        '| Model | Precision ↑ | Recall ↑ | F1 ↑ | Balanced accuracy ↑ | MCC ↑ | Alert fraction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{r[k]:.4f}" for k in [
            'forward_precision', 'forward_recall', 'forward_f1', 'forward_balanced_accuracy', 'forward_mcc', 'forward_alert_fraction']) + ' |')
    lines += ['', 'These are model-specific forward thresholds, not matched staffing budgets. Do not optimise or select thresholds retrospectively from this table.', '',
        '## Matched daily review capacity: nominal top 10%', '',
        '| Model | Precision at capacity ↑ | Coverage / recall at capacity ↑ | Lift ↑ | Actual reviewed fraction |',
        '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{r['capacity'][k]:.4f}" for k in [
            'precision_at_10pct', 'coverage_at_10pct', 'lift_at_10pct', 'selected_fraction_at_10pct']) + ' |')
    lines += ['', 'Ties use expected capture under random tie-breaking. Integer daily capacity rounds up, so actual review fraction slightly exceeds 10%.', '',
        'Full confusion counts, specificity, accuracy, probability RMSE, other capacities and fold metrics are in classification_metric_summary.json.', '',
        'This report adds no significance tests for the secondary metrics. Existing AP tests do not establish significance for F1, coverage or Brier. These are repeatedly inspected development cohorts, not final test results.', '']
    (ROOT / 'CLASSIFICATION_METRIC_COMPARISON.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    run()
