"""Pre-score every June–August replay order for the month-end review page.

Run from root: python -m deployment.scripts.score_replay
Uses the same packaged C3/R1 artifacts that the API serves (see
docs/render_model_equivalence.json). Scores depend only on approval-time
features; outcomes are never read here.
"""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deployment.api.classification_service import ClassificationService
from deployment.api.feature_validation import FeatureValidationError
from deployment.api.model_service import ModelService
import pandas as pd

OUTPUT=ROOT/'deployment/app/data/replay_scores.json'


def score_replay():
    c,r=ClassificationService(),ModelService()
    snapshots=json.loads((ROOT/'deployment/app/data/simulation_orders.json').read_text())
    rows=[row for day in snapshots['days'].values() for row in day]
    if len({row['order_id'] for row in rows})!=len(rows):
        raise SystemExit('Order IDs must be unique across the replay')
    valid,cf,rf,scores=[],[],[],{}
    for row in rows:
        try:
            cf.append(c.validate_features(row['features'])); rf.append(r.validate_features(row['features']))
            valid.append(row)
        except FeatureValidationError:
            scores[row['order_id']]=None  # Same rows the API would reject with 422.
    probabilities=c.probabilities(pd.concat(cf,ignore_index=True))
    days=r.predictions(pd.concat(rf,ignore_index=True))
    for row,p,d in zip(valid,probabilities,days):
        scores[row['order_id']]=[float(p),r.format_prediction(float(d),row['features'])['predicted_days_from_promise']]
    result={'schema_version':1,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'models':{'classification':c.metadata['model_version'],'regression':r.metadata['model_version']},
            'source':'Packaged C3/R1 artifacts, identical to the deployed API per render_model_equivalence.json.',
            'columns':['late_probability','predicted_days_from_promise'],
            'scores':scores}
    OUTPUT.write_text(json.dumps(result,separators=(',',':'))+'\n')
    manifest_path=ROOT/'deployment/artifacts/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['frontend_data'].pop('june_scores.json',None)
    manifest['frontend_data'][OUTPUT.name]=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    manifest['frontend_data']=dict(sorted(manifest['frontend_data'].items()))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(valid)} scored, {len(scores)-len(valid)} rejected -> {OUTPUT.relative_to(ROOT)}')

if __name__=='__main__':score_replay()
