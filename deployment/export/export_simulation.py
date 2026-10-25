"""Export June–August approval snapshots and separate retrospective labels without refitting.

June is the selection month; July and August are frozen-model monitoring months.

Run from repository root: python -m deployment.export.export_simulation
Raw Olist tables are needed only by this offline utility.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_olist_data
from src.features import add_extra_features, build_feature_table
from src.labels import attach_prediction_labels, attach_regression_actuals

ARTIFACTS = ROOT / 'deployment/artifacts'
DEMO_DATA = ROOT / 'deployment/app/data'
FIRST_APPROVAL, LAST_APPROVAL = '2018-06-01', '2018-08-31'


def export():
    data = load_olist_data(verbose=False)
    frame = add_extra_features(build_feature_table(data), data)
    rows = frame.loc[frame.order_approved_dt.between(FIRST_APPROVAL, LAST_APPROVAL)].copy()
    rows = rows.sort_values(['order_approved_dt', 'order_id'])
    metadata = json.loads((ARTIFACTS / 'classification_model_stats.json').read_text())
    columns = metadata['features']['numeric'] + metadata['features']['categorical']
    snapshots = {'schema_version': 1, 'model_available_date': '2018-06-02',
                 'timezone': 'America/Sao_Paulo (source timestamps treated as naive local wall time)',
                 'batch_definition': 'approval_date', 'days': {},
                 'limitations': [
                     'Historical approval features are reconstructed from final item, payment, seller and catalogue snapshots; their change histories are unavailable.',
                     'Cancellation timestamps are unavailable. Final canceled/unavailable status is never used for operational eligibility.',
                     'Regression training and Phase 2 evaluation excluded final canceled/unavailable orders; serving the full approval cohort extends that population.',
                     'Missing categorical values or unsupported features are excluded as invalid model inputs with an explicit reason.',
                     'Source timestamps have no offset; dates use Brazilian local wall time without conversion.',
                 ]}
    for day, group in rows.groupby('order_approved_dt', sort=True):
        result = []
        for _, row in group.iterrows():
            features = row[columns].astype(object).where(row[columns].notna(), None).to_dict()
            def stamp(name):
                value = row[name]
                return None if pd.isna(value) else pd.Timestamp(value).isoformat()
            result.append({'order_id': row.order_id, 'approved_at': stamp('order_approved_at'),
                           'promised_delivery_date': stamp('order_estimated_delivery_date'),
                           'delivered_at': stamp('order_delivered_customer_date'), 'features': features})
        snapshots['days'][day.strftime('%Y-%m-%d')] = result
    labels = attach_prediction_labels(rows[['order_id']].assign(predicted_probability=0.0), data['orders'], lookback=45)
    outcomes = {'schema_version': 1, 'display_only': True,
                'label_policy': 'Phase 2 labels as of approval day + 45 days inclusive; unknown outcomes remain null.',
                'outcomes': {row.order_id: None if pd.isna(row.actual_label) else int(row.actual_label)
                             for row in labels.itertuples()}}
    # R1's own target, for retrospective accuracy only: same 45-day capping and per-cohort
    # evaluation date as Phase 2, excluding final canceled/unavailable orders as R1 evaluation did.
    actuals = attach_regression_actuals(rows[['order_id']], data['orders'], waiting_period=45)
    status = data['orders'].set_index('order_id')['order_status']
    excluded = actuals['order_id'].map(status).isin(['canceled', 'unavailable'])
    timing = {'schema_version': 1, 'display_only': True,
              'target_policy': 'Phase 2 R1 target: min(delivered - approved, 45) - (promised - approved) days, evaluated at approval + 45 days; '
                               'final canceled/unavailable orders excluded as in R1 evaluation; unknown remain null.',
              'actual_days_from_promise': {row.order_id: None if (skip or pd.isna(row.actual_target)) else float(row.actual_target)
                                           for row, skip in zip(actuals.itertuples(), excluded)}}
    for filename, payload in [('simulation_orders.json', snapshots), ('simulation_outcomes.json', outcomes), ('simulation_timing.json', timing)]:
        DEMO_DATA.mkdir(parents=True, exist_ok=True)
        (DEMO_DATA / filename).write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    manifest = {'export': 'python -m deployment.export.export_simulation', 'approval_rows': len(rows),
                'approval_days': len(snapshots['days']), 'model_available_date': snapshots['model_available_date'],
                'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ARTIFACTS.iterdir()) if p.is_file() and p.suffix in {'.joblib', '.json'} and p.name != 'manifest.json'}}
    manifest['frontend_data'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEMO_DATA.glob('*.json'))}
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(rows)} June–August approvals; retrospective outcomes kept separate')

if __name__ == '__main__':
    export()
