"""Integrated C3/R1 inference and historical daily operations API."""
from __future__ import annotations

import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from .classification_service import ARTIFACTS, ClassificationService
from .feature_validation import FeatureValidationError
from .model_service import ModelService
from .simulation import MINIMUM_DATE, SimulationDataError, build_worklist, configuration, retrospective

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


app = FastAPI(title="Team 8 Delivery Operations API", version="2.0.0", lifespan=lifespan,
              description="Approval-time C3 risk and R1 days from promise; daily historical prioritisation.")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any] = Field(description="All approval-time features from /model-info; numeric nulls use fitted medians.")


class BatchRow(PredictionRequest):
    order_id: str | None = Field(default=None, max_length=128)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


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


@app.post("/explain", tags=["prediction"])
def explain(request: PredictionRequest):
    return safe_call(lambda: require_service().explain(request.features))


def artifact_json(name):
    try:
        return json.loads((ARTIFACTS / name).read_text())
    except Exception:
        logger.exception("Companion artifact unavailable: %s", name)
        raise HTTPException(503, "Demonstration data is unavailable") from None


@app.get("/samples", tags=["demonstration"])
def samples():
    return {"model_version": require_service().metadata["model_version"], "samples": artifact_json("classification_sample_orders.json")}


@app.get("/simulation", tags=["demonstration"])
def simulation():
    try:
        return configuration(artifact_json("simulation_orders.json"))
    except SimulationDataError:
        raise HTTPException(503, "Historical simulation data is unavailable") from None


def daily_worklist(simulation_date, review_proportion, minimum_days_remaining):
    data = artifact_json("simulation_orders.json")
    classifier, regressor = require_service(), require_service("regression")
    def validate_row(features):
        classifier.validate_features(features)
        regressor.validate_features(features)
    try:
        return build_worklist(data, simulation_date, review_proportion, minimum_days_remaining,
                              lambda rows: safe_call(lambda: score_rows(rows)), validate_row)
    except SimulationDataError:
        logger.exception("Historical simulation data invalid")
        raise HTTPException(503, "Historical simulation data is unavailable") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/worklist", tags=["demonstration"])
def worklist(simulation_date: date = MINIMUM_DATE, review_proportion: float = Query(.3, ge=0, le=1),
             minimum_days_remaining: int = Query(1, ge=0, le=365)):
    return daily_worklist(simulation_date, review_proportion, minimum_days_remaining)


@app.get("/outcomes", tags=["retrospective"])
def outcomes(simulation_date: date = MINIMUM_DATE, review_proportion: float = Query(.3, ge=0, le=1),
             minimum_days_remaining: int = Query(1, ge=0, le=365)):
    policy = daily_worklist(simulation_date, review_proportion, minimum_days_remaining)
    return retrospective(policy, artifact_json("simulation_outcomes.json")["outcomes"])


@app.get("/logs/summary", tags=["monitoring"])
def logs_summary():
    with monitor_lock:
        entries, count, requests = list(recent_predictions), total_predictions, total_requests
    scores = [entry["score"] for entry in entries]
    return {"request_count": requests, "prediction_count": count, "window_size": len(entries),
            "window_capacity": recent_predictions.maxlen, "mean_score": sum(scores) / len(scores) if scores else None,
            "recent_warnings": [{"timestamp": row["timestamp"], "features": row["warning_features"]} for row in entries if row["warning_features"]][-20:],
            "counting_policy": "Counts direct prediction HTTP calls and successful classification rows; simulation and explanation probes excluded.",
            "storage": "Process-local bounded memory; no order identifiers or feature values retained."}


@app.get("/model-info", tags=["system"])
def model_info():
    return {"classification": require_service().metadata, "regression": require_service("regression").metadata}


@app.get("/model-info/classification", tags=["system"])
def classifier_info():
    return require_service().metadata


@app.get("/model-info/regression", tags=["system"])
def regression_info():
    return require_service("regression").metadata
