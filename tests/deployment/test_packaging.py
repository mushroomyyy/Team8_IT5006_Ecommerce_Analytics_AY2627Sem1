"""Artifact packaging and deployment-isolation checks.

Run from the repository: python -m unittest tests.deployment.test_packaging.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from deployment.api.transforms import ExpandColumns

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "deployment/artifacts"
DEMO_DATA = ROOT / "deployment/app/data"


class PackagingTests(unittest.TestCase):
    def test_linear_artifacts_use_the_standalone_transformer(self):
        for name in ("classification_model.joblib", "regression_model.joblib"):
            with self.subTest(artifact=name):
                bundle = joblib.load(ARTIFACTS / name)
                if bundle["metadata"]["family"] == "linear":
                    expand = bundle["model"].named_steps["preprocessor"].named_steps["expand"]
                    self.assertIs(type(expand), ExpandColumns)

    def test_custom_transformer_parity_for_missing_zero_and_mixed_values(self):
        sys.path.insert(0, str(ROOT))
        try:
            old_class = importlib.import_module("src.linear_transforms").ExpandColumns
        finally:
            sys.path.remove(str(ROOT))
        frame = pd.DataFrame({"amount": [0., 100., np.nan], "order_approved_month": [1, 12, 6],
                              "route_type": ["3. Mixed", "1. All interstate", None]})
        for cyclic in (False, True):
            for merge in (False, True):
                params = dict(num_cols=["amount", "order_approved_month"], cat_cols=["route_type"],
                              log_cols=["amount"], cyclic=cyclic, merge_mixed_route=merge)
                old, new = old_class(**params).fit(frame), ExpandColumns(**params).fit(frame)
                pd.testing.assert_frame_equal(old.transform(frame), new.transform(frame), check_exact=True)
                np.testing.assert_array_equal(old.get_feature_names_out(), new.get_feature_names_out())
                self.assertEqual(old.source_, new.source_)
                bad = frame.copy()
                bad.loc[0, "amount"] = -1
                for transformer in (old, new):
                    with self.assertRaises(ValueError):
                        transformer.transform(bad)

    def test_standalone_inference_does_not_import_phase2(self):
        code = '''
import importlib.abc, sys
class BlockHistoricalCode(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src" or fullname.startswith("src.") or fullname == "phase2" or fullname.startswith("phase2."):
            raise AssertionError("Inference attempted historical import: " + fullname)
sys.meta_path.insert(0, BlockHistoricalCode())
from deployment.api.classification_service import ClassificationService
from deployment.api.model_service import ModelService
import json
from pathlib import Path
sample = json.loads((Path(ClassificationService().artifact_path).parents[1] / "app/data/classification_sample_orders.json").read_text())[0]
sample.pop("order_id", None)
ClassificationService().predict(sample)
ModelService().predict(sample)
assert not any(n == "src" or n.startswith("src.") or n == "phase2" or n.startswith("phase2.") for n in sys.modules)
'''
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable, "-c", code], cwd=directory, env=env,
                                       text=True, capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_simulation_artifacts_separate_labels_and_final_status(self):
        snapshot = json.loads((DEMO_DATA / "simulation_orders.json").read_text())
        outcomes = json.loads((DEMO_DATA / "simulation_outcomes.json").read_text())
        ids = []
        for batch in snapshot["days"].values():
            for order in batch:
                self.assertEqual(set(order), {"order_id", "approved_at", "promised_delivery_date", "delivered_at", "features"})
                self.assertFalse({"order_status", "actual_label", "target", "outcome"}.intersection(order["features"]))
                ids.append(order["order_id"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(outcomes["outcomes"]))
        self.assertTrue(outcomes["display_only"])
        self.assertTrue(all(value in (None, 0, 1, "canceled") for value in outcomes["outcomes"].values()))

    def test_manifest_hashes_match_every_packaged_artifact(self):
        manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
        expected = {p.name for p in ARTIFACTS.iterdir() if p.suffix in {".json", ".joblib"} and p.name != "manifest.json"}
        self.assertEqual(set(manifest["artifacts"]), expected)
        self.assertEqual(set(manifest["frontend_data"]), {p.name for p in DEMO_DATA.glob("*.json")})
        for name, digest in manifest["frontend_data"].items():
            self.assertEqual(hashlib.sha256((DEMO_DATA / name).read_bytes()).hexdigest(), digest, name)
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
