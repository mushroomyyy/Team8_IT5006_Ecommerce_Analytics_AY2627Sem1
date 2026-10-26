"""Retrospective evaluation of the fixed demo daily policy; no policy tuning.

Run from root: python -m deployment.scripts.evaluate_daily
Writes explicit daily and pooled denominators under docs/.
"""
from __future__ import annotations
import json
import sys
from datetime import date,timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deployment.api.classification_service import ClassificationService,ARTIFACTS
from deployment.api.model_service import ModelService
from deployment.app.simulation import build_worklist,retrospective
import pandas as pd


def evaluate():
    c,r=ClassificationService(),ModelService()
    snapshots=json.loads((ROOT/'deployment/app/data/simulation_orders.json').read_text())
    outcomes=json.loads((ROOT/'deployment/app/data/simulation_outcomes.json').read_text())['outcomes']
    def score(batch):
        rows = [item["features"] for item in batch]
        cf=pd.concat([c.validate_features(x) for x in rows],ignore_index=True)
        rf=pd.concat([r.validate_features(x) for x in rows],ignore_index=True)
        return [{'classification':c.format_prediction(float(cv),x),'regression':r.format_prediction(float(rv),x)}
                for x,cv,rv in zip(rows,c.probabilities(cf),r.predictions(rf))]
    days=[]
    pooled={group:{key:0 for key in ['total','known','unknown','late']} for group in ['all','eligible','selected']}
    for n in range(30):
        current=date(2018,6,2)+timedelta(days=n)
        worklist=build_worklist(snapshots,current,.3,1,score)
        metrics=retrospective(worklist,outcomes)['metrics']
        days.append({'simulation_date':str(current),'counts':worklist['counts'],'exclusion_reasons':worklist['exclusion_reasons'],'metrics':metrics})
        for group in pooled:
            for key in pooled[group]:pooled[group][key]+=metrics[group][key]
    def ratio(a,b):return a/b if b else None
    pooled.update(selected_late_rate=ratio(pooled['selected']['late'],pooled['selected']['known']),
                  eligible_late_coverage=ratio(pooled['selected']['late'],pooled['eligible']['late']),
                  all_late_coverage=ratio(pooled['selected']['late'],pooled['all']['late']))
    result={'notice':'Retrospective evaluation of preset daily policy; models were chosen on Validation, so June is held-out test data, but the policy settings are demonstration assumptions.',
            'policy':{'review_proportion':.3,'minimum_days_remaining':1,'rounding':'ceil each day','ranking':f"{c.metadata['model_code']} descending, order ID ascending"},
            'pooled':pooled,'days':days}
    (ROOT/'docs/daily_policy_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pooled,indent=2))

if __name__=='__main__':evaluate()
