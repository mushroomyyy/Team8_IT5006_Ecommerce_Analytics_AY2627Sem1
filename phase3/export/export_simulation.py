"""Export June approval snapshots and separate retrospective labels without refitting.

Run from repository root: python -m phase3.export.export_simulation
Raw Olist tables are needed only by this offline utility.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'phase2'))
from src.data import load_olist_data
from src.features import add_extra_features, build_feature_table
from src.labels import attach_prediction_labels

ARTIFACTS = ROOT / 'phase3/artifacts'


def export():
    data = load_olist_data(verbose=False)
    frame = add_extra_features(build_feature_table(data), data)
    rows = frame.loc[frame.order_approved_dt.between('2018-06-01', '2018-06-30')].copy()
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
    for filename, payload in [('simulation_orders.json', snapshots), ('simulation_outcomes.json', outcomes)]:
        (ARTIFACTS / filename).write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    manifest = {'export': 'python -m phase3.export.export_simulation', 'approval_rows': len(rows),
                'approval_days': len(snapshots['days']), 'model_available_date': snapshots['model_available_date'],
                'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ARTIFACTS.iterdir()) if p.is_file() and p.suffix in {'.joblib', '.json'} and p.name != 'manifest.json'}}
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(rows)} June approvals; retrospective outcomes kept separate')

if __name__ == '__main__':
    export()
