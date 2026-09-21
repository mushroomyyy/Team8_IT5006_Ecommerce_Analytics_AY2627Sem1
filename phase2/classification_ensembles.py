"""Fixed soft-voting ablations; no weights learned from validation outcomes."""
import json
import numpy as np
import pandas as pd
from classification_experiments import ROOT
from classification_backtest import evaluate_out_of_fold_predictions
from classification_significance import compare
from model_workbench import to_builtin


def equal_weight_blend(predictions, members, name):
    if len(set(members)) != len(members) or len(members) < 2:
        raise ValueError('At least two unique members required')
    reference = None
    probabilities = []
    for member in members:
        current = predictions.loc[predictions.model.eq(member)].sort_values('order_id').reset_index(drop=True)
        if current.empty or current.order_id.duplicated().any():
            raise ValueError('Missing or duplicated member predictions')
        if not np.isfinite(current.probability).all() or not current.probability.between(0, 1).all():
            raise ValueError('Invalid probabilities')
        if reference is None:
            reference = current.copy()
        else:
            for col in ['order_id', 'approval_date', 'label_available_at', 'actual', 'fold', 'sample']:
                if not reference[col].equals(current[col]):
                    raise ValueError('Unpaired ensemble inputs: ' + col)
        probabilities.append(current.probability.to_numpy())
    reference['probability'] = np.mean(probabilities, axis=0)
    reference['model'] = name
    # Each ensemble learns its prospective threshold from earlier mature labels.
    return reference.drop(columns='predicted', errors='ignore')


def run():
    frame = pd.read_csv(ROOT / 'classification_time_distance_predictions.csv',
        parse_dates=['approval_date', 'label_available_at'])
    frame = frame.loc[frame.model.str.startswith('purchase_distance_')].copy()
    prior = pd.read_csv(ROOT / 'classification_backtest_predictions.csv',
        parse_dates=['approval_date', 'label_available_at'])
    prior = prior.loc[prior.model.eq('logistic_regression') & prior['sample'].eq('cross_validation')].copy()
    prior['model'] = 'base_logistic_regression'
    frame = pd.concat([frame, prior], ignore_index=True)
    assert frame['sample'].eq('cross_validation').all()
    definitions = {
        'ensemble_logistic_catboost': ['purchase_distance_logistic_regression', 'purchase_distance_catboost'],
        'ensemble_all_four': ['purchase_distance_' + n for n in
            ['logistic_regression', 'random_forest', 'xgboost', 'catboost']],
    }
    for name, members in definitions.items():
        frame = pd.concat([frame, equal_weight_blend(frame, members, name)], ignore_index=True)
    results = dict(status='exploratory_fixed_weight_ensembles', final_holdout_evaluated=False,
        definitions=definitions, weighting='Equal arithmetic mean of probabilities; no label-fitted weights.',
        caveats=['Development-selected feature configurations; not independent final testing.',
            'Class-weighted probabilities not calibrated. Voting may alter ranking without improving probability quality.'], models={})
    evaluated = []
    for name, pred in frame.groupby('model'):
        pred = pred.reset_index(drop=True)
        results['models'][name] = evaluate_out_of_fold_predictions(pred)
        evaluated.append(pred)
    frame = pd.concat(evaluated, ignore_index=True)
    (ROOT / 'classification_ensemble_results.json').write_text(json.dumps(to_builtin(results), indent=2, allow_nan=False))
    frame.to_csv(ROOT / 'classification_ensemble_predictions.csv', index=False)
    stats = dict(status='exploratory_conditional_not_final',
        comparison_family='Six candidates vs original logistic; Holm within each block length.',
        limitations='Fixed predictions and months; no retraining or model-search uncertainty. Marginal intervals. See original statistical methodology.', analyses=[])
    for days in [7, 3, 14]:
        print('Ensemble bootstrap block:', days, flush=True)
        stats['analyses'].append(compare(frame, repeats=1999, block_days=days))
    (ROOT / 'classification_ensemble_significance.json').write_text(json.dumps(stats, indent=2, allow_nan=False))


if __name__ == '__main__':
    run()
