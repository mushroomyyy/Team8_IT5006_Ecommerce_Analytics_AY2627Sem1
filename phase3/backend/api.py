"""Integrated C3/R1 inference service for approval-time feature rows."""
from __future__ import annotations

import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .classification_service import ClassificationService
from .feature_validation import FeatureValidationError
from .model_service import ModelService

logger = logging.getLogger(__name__)
service = None
regression_service = None
recent_predictions = deque(maxlen=1000)
monitor_lock = Lock()
total_predictions = total_requests = 0


@asynccontextmanager
async def lifespan(app):
    global service, regression_service, total_predictions, total_requests
    service = regression_service = None
    with monitor_lock:
        recent_predictions.clear()
        total_predictions = total_requests = 0
    for factory, name in ((ClassificationService, "classification"), (ModelService, "regression")):
        try:
            loaded = factory()
            if name == "classification":
                service = loaded
            else:
                regression_service = loaded
        except Exception:
            logger.exception("%s model could not be loaded", name)
    yield
    service = regression_service = None


app = FastAPI(title="Team 8 Delivery Operations API", version="2.1.0", lifespan=lifespan,
              description="Approval-time C3 late-delivery risk and R1 days from promised delivery date.")


# Invented order used to prefill Swagger "Try it out" (no real transaction values);
# matches docs/synthetic_prediction.json and scripts/verify_http.py.
EXAMPLE_FEATURES = {
    "order_approved_day_of_week": 5, "order_approved_day_of_month": 1, "order_approved_month": 6,
    "order_item_count": 1, "order_seller_count": 1, "order_frieght_value_sum": 12.5,
    "order_product_weight_g_sum": 800.0, "order_product_category_count": 1,
    "seller_dist_km_mean": 350.0, "seller_state_count": 1, "payment_value_sum": 100.0,
    "promised_lead_days": 20, "approval_lag_hours": 2.0, "order_product_volume_cm3_sum": 1000.0,
    "order_product_photos_mean": 3.0, "freight_to_price_ratio": 0.12, "payment_installments_max": 1,
    "is_weekend_approval": 0, "is_black_friday_period": 0, "is_december_peak": 0,
    "orders_approved_prev_7d": 1000, "customer_state": "SP",
    "route_type": "1. All interstate", "payment_combination": "credit_card",
}
EXAMPLE_FEATURES_2 = {**EXAMPLE_FEATURES, "customer_state": "RJ", "promised_lead_days": 8,
                      "seller_dist_km_mean": 900.0, "order_frieght_value_sum": 30.0, "freight_to_price_ratio": 0.3}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"features": EXAMPLE_FEATURES}]})
    features: dict[str, Any] = Field(description="All approval-time features from /model-info; numeric nulls use fitted medians.")


class BatchRow(PredictionRequest):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"order_id": "example-1", "features": EXAMPLE_FEATURES}]})
    order_id: str | None = Field(default=None, max_length=128)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"rows": [
        {"order_id": "example-1", "features": EXAMPLE_FEATURES},
        {"order_id": "example-2", "features": EXAMPLE_FEATURES_2}]}]})
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000,
                                       description="1–1,000 objects, each {\"order_id\": optional string, \"features\": {...}}.")


def require_service(name="classification"):
    loaded = service if name == "classification" else regression_service
    if loaded is None:
        raise HTTPException(503, f"{name.capitalize()} model is not loaded")
    return loaded


def record_request():
    global total_requests
    with monitor_lock:
        total_requests += 1


def record_prediction(result):
    global total_predictions
    result = result.get("classification", result)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "score": result["late_probability"],
             "warning_features": [item["feature"] for item in result.get("drift_warnings", [])]}
    with monitor_lock:
        total_predictions += 1
        recent_predictions.append(entry)


def safe_call(callback):
    try:
        return callback()
    except FeatureValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference operation failed")
        raise HTTPException(500, "Prediction failed; please try again or contact the service owner") from None


def score_rows(rows):
    classifier, regressor = require_service(), require_service("regression")
    if not rows:
        return []
    cframe = pd.concat([classifier.validate_features(row) for row in rows], ignore_index=True)
    rframe = pd.concat([regressor.validate_features(row) for row in rows], ignore_index=True)
    cscores, rscores = classifier.probabilities(cframe), regressor.predictions(rframe)
    return [{"classification": classifier.format_prediction(float(c), row),
             "regression": regressor.format_prediction(float(r), row)} for row, c, r in zip(rows, cscores, rscores)]


@app.get("/", tags=["system"])
def root():
    return {"service": "integrated delivery operations", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def health():
    c, r = require_service(), require_service("regression")
    return {"status": "ok", "model_loaded": True, "models": {"classification": c.metadata["model_version"], "regression": r.metadata["model_version"]}}


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest):
    record_request()
    result = safe_call(lambda: score_rows([request.features])[0])
    record_prediction(result)
    return result


@app.post("/predict/classification", tags=["prediction"])
def predict_classification(request: PredictionRequest):
    record_request()
    result = safe_call(lambda: require_service().predict(request.features))
    record_prediction(result)
    return result


@app.post("/predict/regression", tags=["prediction"])
def predict_regression(request: PredictionRequest):
    record_request()
    return safe_call(lambda: require_service("regression").predict(request.features))


@app.post("/predict/batch", tags=["prediction"])
def predict_batch(request: BatchRequest):
    c, r = require_service(), require_service("regression")
    record_request()
    results, valid = [], []
    for index, row in enumerate(request.rows):
        item = {"index": index}
        try:
            parsed = BatchRow.model_validate(row)
            item["order_id"] = parsed.order_id
            c.validate_features(parsed.features)
            r.validate_features(parsed.features)
            valid.append((item, parsed.features))
        except FeatureValidationError as exc:
            item.update(error=str(exc), status_code=422)
        except Exception:
            item.update(error="Each row must contain features and an optional order_id", status_code=422)
        results.append(item)
    if valid:
        try:
            predictions = score_rows([features for _, features in valid])
            for (item, _), prediction in zip(valid, predictions):
                item["prediction"] = prediction
                record_prediction(prediction)
        except Exception:
            logger.exception("Batch inference failed")
            for item, _ in valid:
                item.update(error="Batch prediction failed; please try again or contact the service owner", status_code=500)
    return {"results": results, "successful_rows": sum("prediction" in row for row in results)}


@app.get("/logs/summary", tags=["monitoring"])
def logs_summary():
    with monitor_lock:
        entries, count, requests = list(recent_predictions), total_predictions, total_requests
    scores = [entry["score"] for entry in entries]
    return {"request_count": requests, "prediction_count": count, "window_size": len(entries),
            "window_capacity": recent_predictions.maxlen, "mean_score": sum(scores) / len(scores) if scores else None,
            "recent_warnings": [{"timestamp": row["timestamp"], "features": row["warning_features"]} for row in entries if row["warning_features"]][-20:],
            "counting_policy": "Counts direct prediction HTTP calls and successful classification rows.",
            "storage": "Process-local bounded memory; no order identifiers or feature values retained."}


# Fitted bundle metadata is preserved for reproducibility. HTTP metadata
# describes inference rather than historical demo distributions or old policies.
OMITTED_METADATA_FIELDS = frozenset({
    "threshold", "threshold_policy", "june_score_distribution", "global_importance", "baseline_features",
})


def inference_metadata(loaded):
    return {key: value for key, value in loaded.metadata.items() if key not in OMITTED_METADATA_FIELDS}


@app.get("/model-info", tags=["system"])
def model_info():
    return {"classification": inference_metadata(require_service()),
            "regression": inference_metadata(require_service("regression"))}


@app.get("/model-info/classification", tags=["system"])
def classifier_info():
    return inference_metadata(require_service())


@app.get("/model-info/regression", tags=["system"])
def regression_info():
    return inference_metadata(require_service("regression"))
