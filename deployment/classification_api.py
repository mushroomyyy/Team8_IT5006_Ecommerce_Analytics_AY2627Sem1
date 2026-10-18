"""FastAPI service for the packaged C3 late-delivery classifier."""

from __future__ import annotations

import logging
import json
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .classification_service import ARTIFACTS, ClassificationService
from .feature_validation import FeatureValidationError


logger = logging.getLogger(__name__)
service: ClassificationService | None = None
recent_predictions = deque(maxlen=1000)
monitor_lock = Lock()
total_predictions = 0
total_requests = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, total_predictions, total_requests
    service = None
    with monitor_lock:
        recent_predictions.clear()
        total_predictions = 0
        total_requests = 0
    try:
        service = ClassificationService()
    except Exception:
        # Keep health observable and predictions unavailable if loading fails.
        logger.exception("Classification model could not be loaded")
    yield
    service = None


app = FastAPI(
    title="Team 8 Late Delivery Classification API",
    version="1.0.0",
    description="Predict late-delivery probability from approval-time order features.",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any] = Field(
        ...,
        description="All approval-time features listed by /model-info. Numeric nulls use fitted medians; categorical values must exactly match /model-info categories.",
        examples=[{
            "order_approved_day_of_week": 2,
            "order_approved_day_of_month": 15,
            "order_approved_month": 6,
            "order_item_count": 1,
            "order_seller_count": 1,
            "order_frieght_value_sum": 12.5,
            "order_product_weight_g_sum": 800.0,
            "order_product_category_count": 1,
            "seller_dist_km_mean": 350.0,
            "seller_state_count": 1,
            "payment_value_sum": 100.0,
            "promised_lead_days": 20,
            "approval_lag_hours": 2.0,
            "order_product_volume_cm3_sum": 1000.0,
            "order_product_photos_mean": 3.0,
            "freight_to_price_ratio": 0.12,
            "payment_installments_max": 1,
            "is_weekend_approval": 0,
            "is_black_friday_period": 0,
            "is_december_peak": 0,
            "orders_approved_prev_7d": 1000,
            "customer_state": "SP",
            "route_type": "1. All interstate",
            "payment_combination": "credit_card",
        }],
    )


def require_service() -> ClassificationService:
    if service is None:
        raise HTTPException(status_code=503, detail="Classification model is not loaded")
    return service


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"service": "late-delivery classification", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    loaded = require_service()
    return {"status": "ok", "model_loaded": True,
            "model_code": loaded.metadata["model_code"],
            "model_version": loaded.metadata["model_version"]}


@app.post("/predict", tags=["prediction"])
@app.post("/predict/classification", tags=["prediction"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    loaded = require_service()
    record_request()
    return run_prediction(loaded, request.features)


def record_request():
    global total_requests
    with monitor_lock:
        total_requests += 1


def record_prediction(result):
    global total_predictions
    # Never retain order identifiers, features, raw requests or personal data.
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "score": result["late_probability"],
             "above_watch_list_boundary": result.get("above_watch_list_boundary"),
             "warning_features": [item["feature"] for item in result.get("drift_warnings", [])]}
    with monitor_lock:
        total_predictions += 1
        recent_predictions.append(entry)


def run_prediction(loaded, features):
    try:
        result = loaded.predict(features)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Classification inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed; please try again or contact the service owner") from None
    record_prediction(result)
    return result


class BatchRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]
    order_id: str | None = Field(default=None, max_length=128)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


@app.post("/predict/batch", tags=["prediction"])
def predict_batch(request: BatchRequest):
    loaded = require_service()
    record_request()
    results, frames, valid = [], [], []
    for index, row in enumerate(request.rows):
        try:
            parsed = BatchRow.model_validate(row)
        except Exception:
            results.append({"index": index, "error": "Each row must contain features and an optional order_id", "status_code": 422})
            continue
        item = {"index": index, "order_id": parsed.order_id}
        try:
            frames.append(loaded.validate_features(parsed.features))
            valid.append((item, parsed.features))
        except FeatureValidationError as exc:
            item.update(error=str(exc), status_code=422)
        results.append(item)
    if valid:
        try:
            scores = loaded.probabilities(pd.concat(frames, ignore_index=True))
            for (item, features), score in zip(valid, scores):
                prediction = loaded.format_prediction(float(score), features)
                item["prediction"] = prediction
                record_prediction(prediction)
        except Exception:
            logger.exception("Classification batch inference failed")
            for item, _ in valid:
                item.update(error="Batch prediction failed; please try again or contact the service owner", status_code=500)
    return {"model_version": loaded.metadata["model_version"], "results": results,
            "successful_rows": sum("prediction" in row for row in results)}


@app.post("/explain", tags=["prediction"])
def explain(request: PredictionRequest):
    loaded = require_service()
    try:
        return loaded.explain(request.features)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Classification explanation failed")
        raise HTTPException(status_code=500, detail="Explanation failed; please try again or contact the service owner") from None


def artifact_json(name):
    loaded = require_service()
    try:
        data = json.loads((ARTIFACTS / name).read_text())
        if isinstance(data, dict) and data.get("model_version") != loaded.metadata["model_version"]:
            raise ValueError("Companion artifact version mismatch")
        return data
    except Exception:
        logger.exception("Classification companion artifact unavailable")
        raise HTTPException(status_code=503, detail="Demonstration data is unavailable") from None


@app.get("/samples", tags=["demonstration"])
def samples():
    loaded = require_service()
    return {"model_version": loaded.metadata["model_version"],
            "samples": artifact_json("classification_sample_orders.json")}


@app.get("/replay", tags=["demonstration"])
def replay(day: str | None = None):
    data = artifact_json("classification_june_orders.json")
    days = data["days"]
    if day is not None and day not in days:
        raise HTTPException(status_code=422, detail="day must be an available June 2018 date")
    return {"model_version": data["model_version"], "available_days": sorted(days),
            "orders": days[day] if day else [], "day": day}


@app.get("/outcomes", tags=["demonstration"])
def outcomes(day: str | None = None):
    data = artifact_json("classification_june_outcomes.json")
    selected = data["outcomes"]
    if day is not None:
        ids = {row["order_id"] for row in replay(day)["orders"]}
        selected = {key: value for key, value in selected.items() if key in ids}
    return {"model_version": data["model_version"], "display_only": True, "outcomes": selected,
            "notice": "Historical outcomes for display only; never used to fit, tune or select scores or samples."}


@app.get("/logs/summary", tags=["monitoring"])
def logs_summary():
    loaded = require_service()
    with monitor_lock:
        entries, count, requests = list(recent_predictions), total_predictions, total_requests
    scores = [entry["score"] for entry in entries]
    above = [entry["above_watch_list_boundary"] for entry in entries if entry["above_watch_list_boundary"] is not None]
    return {"model_version": loaded.metadata["model_version"], "request_count": requests, "prediction_count": count,
            "window_size": len(entries), "window_capacity": recent_predictions.maxlen,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "share_above_watch_list_boundary": sum(above) / len(above) if above else None,
            "recent_warnings": [{"timestamp": item["timestamp"], "features": item["warning_features"]}
                                for item in entries if item["warning_features"]][-20:],
            "counting_policy": "request_count counts /predict and /predict/batch HTTP calls; prediction_count counts successful scored rows. Explanation probes excluded.",
            "storage": "Process-local bounded memory; restarts reset counts; no order IDs or feature values retained."}


@app.get("/model-info", tags=["system"])
@app.get("/model-info/classification", tags=["system"])
def model_info() -> dict[str, Any]:
    return require_service().metadata
