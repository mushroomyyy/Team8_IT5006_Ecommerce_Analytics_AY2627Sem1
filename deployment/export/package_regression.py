"""Rebind R1's fitted custom transformer to a standalone module, without refitting.

Default source is the original regression artifact at a fixed historical commit.
Supply a trusted local artifact path as the first argument; never load untrusted pickle files.
"""
from __future__ import annotations
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'deployment/artifacts'
sys.path.insert(0, str(ROOT))
from deployment.api.transforms import ExpandColumns


def package(source=None):
    raw = Path(source).read_bytes() if source else subprocess.check_output(
        ['git', 'show', 'bfea55986f266b4208a8c58f932bf67666391b92:deployment/artifacts/regression_r1_ols.joblib'], cwd=ROOT)
    bundle = joblib.load(io.BytesIO(raw))
    replay = json.loads((ROOT / 'deployment/app/data/classification_june_orders.json').read_text())
    features = bundle['metadata']['features']
    columns = features['numeric'] + features['categorical']
    frame = pd.DataFrame([row for daily in replay['days'].values() for row in daily])[columns]
    before = bundle['model'].predict(frame)
    expand = bundle['model'].named_steps['preprocessor'].named_steps['expand']
    expand.__class__ = ExpandColumns
    after = bundle['model'].predict(frame)
    np.testing.assert_array_equal(before, after)
    path = ARTIFACTS / 'regression_r1_ols.joblib'
    joblib.dump(bundle, path, compress=3)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(before, loaded['model'].predict(frame))
    provenance = {'source_branch': 'origin/model-deployment-r1',
                  'source_commit': 'bfea55986f266b4208a8c58f932bf67666391b92',
                  'source_artifact_sha256': hashlib.sha256(raw).hexdigest(),
                  'packaged_artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'change': 'Only custom transformer class import path rebound to deployment.api.transforms. No fitting, parameter or preprocessing changes.',
                  'equivalence': 'Exact array equality before/after rebinding and after serialization',
                  'verified_rows': len(frame),
                  'runtime_class': 'deployment.api.transforms.ExpandColumns'}
    (ARTIFACTS / 'regression_packaging.json').write_text(json.dumps(provenance, indent=2)+'\n')
    manifest_path = ARTIFACTS / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in manifest['artifacts']:
        manifest['artifacts'][name] = hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged standalone R1; exact equality on {len(frame)} June orders')

if __name__ == '__main__':
    package(sys.argv[1] if len(sys.argv)>1 else None)
