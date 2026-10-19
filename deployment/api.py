"""FastAPI backend for the Phase 2 regression model."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_service import ModelService


app = FastAPI(
    title="Team 8 E-commerce Delivery Prediction API",
    version="1.0.0",
    description="Predicts days before or after the promised delivery date.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
service: ModelService | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="The 24 approval-time model features. See /model-info for the exact feature lists.",
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


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"service": "e-commerce delivery prediction", "docs": "/docs", "health": "/health"}


@app.on_event("startup")
def load_model() -> None:
    global service
    service = ModelService()


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": service is not None,
            "model_code": service.metadata["model_code"] if service else None}


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        return service.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/model-info", tags=["system"])
def model_info() -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return service.metadata
