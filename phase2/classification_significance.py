"""Paired, month-stratified circular moving-block bootstrap of macro AP.

Conditional on saved fitted models. Does not estimate retraining uncertainty,
correct model-search bias, or establish generalisation to an unseen period.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def weighted_ap(y, scores, day_index):
    order = np.argsort(-scores, kind='stable')
    y, scores, day_index = y[order], scores[order], day_index[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]

    def evaluate(day_weights):
        w = day_weights[day_index]
        positives = np.cumsum(w * y)[ends]
        totals = np.cumsum(w)[ends]
        increments = np.diff(np.r_[0., positives])
        if positives[-1] == 0:
            return 0.
        return float(np.sum(increments * np.divide(positives, totals,
            out=np.zeros_like(positives), where=totals > 0)) / positives[-1])
    return evaluate


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        np.asarray(pvalues)[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted.tolist()


def compare(frame, repeats=1999, block_days=7):
    if frame.duplicated(['model', 'order_id']).any():
        raise ValueError('Duplicate model/order predictions')
    names = sorted(frame.model.unique())
    baseline = 'base_logistic_regression'
    fold_evaluators = []
    for _, group in frame.groupby('fold', sort=True):
        ref = group.loc[group.model.eq(baseline)].sort_values('order_id')
        days = pd.date_range(ref.approval_date.min(), ref.approval_date.max())
        day_index = days.get_indexer(ref.approval_date)
        evaluators = []
        for name in names:
            pred = group.loc[group.model.eq(name)].sort_values('order_id')
            for col in ['order_id', 'actual', 'approval_date']:
                if not np.array_equal(ref[col].to_numpy(), pred[col].to_numpy()):
                    raise ValueError(f'Unpaired predictions: {name} / {col}')
            evaluators.append(weighted_ap(ref.actual.to_numpy(), pred.probability.to_numpy(), day_index))
        fold_evaluators.append((len(days), evaluators))
    observed = np.mean([[fn(np.ones(n)) for fn in fns] for n, fns in fold_evaluators], axis=0)
    rng = np.random.default_rng(42)
    bootstrap = np.zeros((repeats, len(names)))
    for b in range(repeats):
        for n, fns in fold_evaluators:
            starts = rng.integers(0, n, size=int(np.ceil(n / block_days)))
            days = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
            weights = np.bincount(days, minlength=n).astype(float)
            bootstrap[b] += [fn(weights) for fn in fns]
    bootstrap /= len(fold_evaluators)
    pairs = [(n, baseline) for n in names if n != baseline]
    pairs += [('enriched_' + n, 'base_' + n) for n in ['random_forest', 'xgboost', 'catboost']
              if 'enriched_' + n in names and 'base_' + n in names]
    rows = []
    for candidate, reference in pairs:
        i, j = names.index(candidate), names.index(reference)
        effect = observed[i] - observed[j]
        draws = bootstrap[:, i] - bootstrap[:, j]
        # Recenter bootstrap differences at the null, not tail probability
        # from the uncentered alternative distribution.
        p = (1 + np.sum(np.abs(draws - effect) >= abs(effect))) / (repeats + 1)
        rows.append(dict(candidate=candidate, reference=reference, delta_macro_ap=float(effect),
            ci95_percentile=np.quantile(draws, [.025, .975]).tolist(), p_approx_two_sided=float(p)))
    for row, adjusted in zip(rows, holm([r['p_approx_two_sided'] for r in rows])):
        row['p_holm'] = adjusted
    return dict(block_days=block_days, repeats=repeats, metric='equal-month mean average precision',
                macro_ap=dict(zip(names, observed.tolist())), comparisons=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, default=Path(__file__).with_name('classification_experiment_predictions.csv'))
    parser.add_argument('--repeats', type=int, default=1999)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, parse_dates=['approval_date'])
    frame = frame.loc[frame['sample'].eq('cross_validation')].copy()
    results = {'status': 'exploratory_conditional_inference_not_final',
        'method': 'Paired circular day-block resampling within each validation month; centered-null approximate p-values; Holm within each block-length analysis.',
        'limitations': 'Fixed predictions, no retraining. Assumes local dependence adequately captured by block length; monthly drift and adaptive feature/model search are not fully accounted for. CIs are marginal, not simultaneous.',
        'analyses': []}
    for block in [7, 3, 14]:
        print('Bootstrap block days:', block, flush=True)
        results['analyses'].append(compare(frame, args.repeats, block))
    args.predictions.with_name('classification_significance_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
