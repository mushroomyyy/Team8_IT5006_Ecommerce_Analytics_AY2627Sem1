"""Independent artifact equivalence and deployment-isolation checks.

Run from the repository: python -m unittest tests.deployment.test_packaging.
Original-artifact comparisons require Git history. The preservation-tar check
uses the handoff backup when available and otherwise verifies its recorded hash.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
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
    @classmethod
    def setUpClass(cls):
        cls.replay = json.loads((DEMO_DATA / "classification_june_orders.json").read_text())
        cls.frame = pd.DataFrame([row for day in cls.replay["days"].values() for row in day])
        cls.packaging = json.loads((ARTIFACTS / "regression_packaging.json").read_text())

    def test_original_regression_exact_on_every_june_order(self):
        original_bytes = subprocess.check_output([
            "git", "show", "bfea55986f266b4208a8c58f932bf67666391b92:deployment/artifacts/regression_r1_ols.joblib"
        ], cwd=ROOT)
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), self.packaging["source_artifact_sha256"])
        # Only this historical-artifact check needs the old custom class module.
        sys.path.insert(0, str(ROOT))
        try:
            original = joblib.load(io.BytesIO(original_bytes))
        finally:
            sys.path.remove(str(ROOT))
        packaged = joblib.load(ARTIFACTS / "regression_r1_ols.joblib")
        cols = original["metadata"]["features"]
        frame = self.frame[cols["numeric"] + cols["categorical"]]
        self.assertEqual(len(frame), 6164)
        np.testing.assert_array_equal(original["model"].predict(frame), packaged["model"].predict(frame))
        self.assertEqual(original["metadata"], packaged["metadata"])
        # Preserve fitted state and documented inputs as well as the full June replay.
        for attribute in ("coef_", "intercept_", "singular_", "rank_", "n_features_in_"):
            np.testing.assert_array_equal(
                getattr(original["model"].named_steps["model"], attribute),
                getattr(packaged["model"].named_steps["model"], attribute))
        examples = {
            "batch": pd.read_csv(ROOT / "docs/examples/batch_example.csv", float_precision="round_trip"),
            "synthetic": pd.DataFrame([json.loads((ROOT / "docs/synthetic_prediction.json").read_text())["features"]]),
            "samples": pd.DataFrame(json.loads((DEMO_DATA / "classification_sample_orders.json").read_text())),
        }
        for name, example in examples.items():
            with self.subTest(example=name):
                features = example[cols["numeric"] + cols["categorical"]]
                np.testing.assert_array_equal(original["model"].predict(features), packaged["model"].predict(features))
        self.assertEqual(hashlib.sha256((ARTIFACTS / "regression_r1_ols.joblib").read_bytes()).hexdigest(),
                         self.packaging["packaged_artifact_sha256"])
        self.assertIs(type(packaged["model"].named_steps["preprocessor"].named_steps["expand"]), ExpandColumns)

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

    def test_classification_binary_preserved(self):
        current = (ARTIFACTS / "classification_c3_rf.joblib").read_bytes()
        self.assertEqual(hashlib.sha256(current).hexdigest(),
                         "17577aa8c96c0f456b55ab8f7417bc5f2916d7f698e5a15c35926743ceb42682")
        backup = Path("/private/tmp/phase3-handoff-preservation/working-files.tar.gz")
        if backup.exists():
            with tarfile.open(backup) as archive:
                self.assertEqual(current, archive.extractfile("deployment/artifacts/classification_c3_rf.joblib").read())

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
        self.assertTrue(all(value in (None, 0, 1) for value in outcomes["outcomes"].values()))

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
