"""FastAPI service for the packaged C3 late-delivery classifier."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .classification_service import ClassificationService


logger = logging.getLogger(__name__)
service: ClassificationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = None
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
        description="All approval-time features listed by /model-info. Use null for missing values.",
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
    try:
        return loaded.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/model-info", tags=["system"])
@app.get("/model-info/classification", tags=["system"])
def model_info() -> dict[str, Any]:
    return require_service().metadata
