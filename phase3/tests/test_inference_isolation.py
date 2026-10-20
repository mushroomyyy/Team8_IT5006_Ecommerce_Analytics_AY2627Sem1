"""Exercise the API with only fitted models, without historical or UI files."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class InferenceIsolationTests(unittest.TestCase):
    def test_api_operates_without_frontend_demo_or_training_code(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            package = isolated / "phase3"
            package.mkdir()
            shutil.copy(root / "phase3/__init__.py", package / "__init__.py")
            shutil.copytree(root / "phase3/backend", package / "backend",
                            ignore=shutil.ignore_patterns("__pycache__"))
            artifacts = package / "artifacts"
            artifacts.mkdir()
            for name in ("classification_c3_rf.joblib", "classification_model_stats.json",
                         "regression_r1_ols.joblib", "regression_model_stats.json"):
                shutil.copy(root / "phase3/artifacts" / name, artifacts / name)
            code = '''
import importlib.abc, math, sys
class RejectApplicationAndTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in ("phase3.frontend", "phase2", "src")):
            raise AssertionError("API depends on application/training code: " + fullname)
sys.meta_path.insert(0, RejectApplicationAndTraining())
from fastapi.testclient import TestClient
from phase3.backend.api import app
with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json()["model_loaded"] is True
    metadata = client.get("/model-info").json()["classification"]
    assert "baseline_features" not in metadata
    assert "june_score_distribution" not in metadata
    features = {
        "order_approved_day_of_week": 5, "order_approved_day_of_month": 1, "order_approved_month": 6,
        "order_item_count": 1, "order_seller_count": 1, "order_frieght_value_sum": 12.5,
        "order_product_weight_g_sum": 800., "order_product_category_count": 1,
        "seller_dist_km_mean": 350., "seller_state_count": 1, "payment_value_sum": 100.,
        "promised_lead_days": 20, "approval_lag_hours": 2., "order_product_volume_cm3_sum": 1000.,
        "order_product_photos_mean": 3., "freight_to_price_ratio": .12, "payment_installments_max": 1,
        "is_weekend_approval": 0, "is_black_friday_period": 0, "is_december_peak": 0,
        "orders_approved_prev_7d": 1000, "customer_state": "SP",
        "route_type": "1. All interstate", "payment_combination": "credit_card",
    }
    single = client.post("/predict", json={"features": features})
    assert single.status_code == 200, single.text
    result = single.json()
    assert 0 <= result["classification"]["late_probability"] <= 1
    assert math.isfinite(result["regression"]["predicted_days_from_promise"])
    batch = client.post("/predict/batch", json={"rows": [{"features": features}]})
    assert batch.status_code == 200, batch.text
    assert batch.json()["successful_rows"] == 1
    for path in ("/samples", "/simulation", "/worklist", "/outcomes"):
        assert client.get(path).status_code == 404, path
    assert client.post("/explain", json={"features": features}).status_code == 404
    assert not any(n == "phase3.frontend" or n.startswith("phase3.frontend.") for n in sys.modules)
print("Standalone inference works with four fitted-model files and no frontend/demo/training tree")
'''
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=isolated,
                                    env={**os.environ, "PYTHONPATH": str(isolated)},
                                    text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
