# Calling the live model API

The Team 8 Delivery Operations API serves the two Phase 2 models over HTTPS:

- **C2B** (logistic regression, 13 BIC-selected features): late-delivery risk, used to
  **rank** orders. `/model-info` also gives its operating cut-off (`threshold`, the
  F1-maximising probability on Validation) and `threshold_policy`.
- **R2B** (OLS, 17 BIC-selected features): predicted days from the promised delivery
  date (negative = early), bounded to a 0-45 day lead time.

> **Local verification only.** The models above passed Docker and HTTP checks on
> 26-27 October 2026. The live Render service is redeployed after the merge to
> `main`; until that is re-verified it may still serve the previous C3/R1 versions.
> Check `/health` for the model versions actually being served. The response
> examples below were captured from the packaged models run locally on 27 October 2026.

| | |
|---|---|
| Base URL | https://team8-delivery-api.onrender.com |
| Interactive docs (Swagger, with prefilled examples) | https://team8-delivery-api.onrender.com/docs |
| Machine-readable schema | https://team8-delivery-api.onrender.com/openapi.json |
| Authentication | None (public demonstration service; no keys or credentials) |
| Format | JSON in, JSON out (`Content-Type: application/json`) |

> **Cold start.** The free Render instance sleeps after about 15 minutes without
> traffic. The first request then takes roughly 30–60 seconds while it wakes;
> later requests are fast. Call `/health` first and allow a generous timeout.

The quickest test is the Swagger page: open an endpoint, select **Try it out**,
then **Execute**. Every prediction endpoint is prefilled with a valid example order.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status and loaded model versions; 503 if a model failed to load |
| GET | `/model-info` | Feature schema, allowed categories, selected features and metadata for both models, including C2B's `threshold` and `threshold_policy` |
| GET | `/model-info/classification`, `/model-info/regression` | The same, for one model |
| POST | `/predict` | C2B and R2B for one order |
| POST | `/predict/batch` | C2B and R2B for 1–1,000 orders in one call |
| POST | `/predict/classification`, `/predict/regression` | One model for one order |
| GET | `/logs/summary` | Live monitoring: request and prediction counts, mean risk, recent drift warnings |

## The input: 24 approval-time features

Every prediction request carries a `features` object with **exactly** these 24
keys, all known when payment is approved. `GET /model-info` lists them and the
allowed category values. Each model uses its own selected subset (`selected_features`
in `/model-info`) and ignores the rest, but all 24 must still be sent and valid.

| Feature | Type | Rule |
|---|---|---|
| `order_approved_day_of_week` | integer | 1 (Monday) to 7 (Sunday) |
| `order_approved_day_of_month` | integer | 1–31, valid for the month |
| `order_approved_month` | integer | 1–12 |
| `order_item_count`, `order_seller_count`, `order_product_category_count`, `seller_state_count`, `payment_installments_max` | integer | ≥ 1 |
| `orders_approved_prev_7d` | integer | ≥ 0 |
| `promised_lead_days` | integer | may be negative |
| `order_frieght_value_sum`, `order_product_weight_g_sum`, `order_product_volume_cm3_sum`, `order_product_photos_mean`, `seller_dist_km_mean`, `payment_value_sum`, `approval_lag_hours`, `freight_to_price_ratio` | number | ≥ 0 |
| `is_weekend_approval`, `is_december_peak`, `is_black_friday_period` | 0 or 1 | the weekend and December flags must match the weekday and month |
| `customer_state`, `route_type`, `payment_combination` | string | must exactly match a category from `/model-info` |

Numeric features may be `null` (the model fills them with its training median);
at least one numeric value is required. Categorical features may not be null.
The feature name `order_frieght_value_sum` keeps the training data's spelling.

A complete, invented example order (no real transaction data) is in
[`synthetic_prediction.json`](synthetic_prediction.json).

## Examples

Responses below were captured on 27 October 2026 from the packaged C2B/R2B models
run locally (not from the live service), using the invented example order.

### Check the service

```bash
curl https://team8-delivery-api.onrender.com/health
```

```json
{"status": "ok", "model_loaded": true,
 "models": {"classification": "phase2-c2b-logistic-2018-06-02", "regression": "phase2-r2b-ols-2018-06-02"}}
```

### Score one order

```bash
curl -X POST https://team8-delivery-api.onrender.com/predict \
  -H 'Content-Type: application/json' \
  --data-binary @docs/synthetic_prediction.json
```

```json
{
  "classification": {
    "late_probability": 0.05184186498640409,
    "drift_warnings": [],
    "score_interpretation": "Estimated late-delivery risk; not a calibrated guarantee.",
    "model_code": "C2B",
    "model_version": "phase2-c2b-logistic-2018-06-02"
  },
  "regression": {
    "predicted_days_from_promise": -9.88466003218192,
    "raw_days_from_promise": -9.88466003218192,
    "late_by_days": 0.0,
    "early_by_days": 9.88466003218192,
    "clipping_applied": true,
    "clipping_policy": "Predicted approval-to-delivery duration bounded to 0–45 days, matching Phase 2 evaluation.",
    "target": "days from promised delivery date; negative means early, positive means late",
    "model_code": "R2B",
    "model_version": "phase2-r2b-ols-2018-06-02"
  }
}
```

How to read it:

| Field | Meaning |
|---|---|
| `classification.late_probability` | C2B late-delivery risk score, 0–1. Use it to rank orders; it is not a calibrated probability. Compare it with `threshold` from `/model-info` (0.1282) for a flag; the response itself carries no flag. |
| `classification.drift_warnings` | Inputs outside the training 1st–99th percentile (empty when none) |
| `regression.predicted_days_from_promise` | R2B days from the promised date: negative = early, positive = late. Approval-to-delivery is bounded to a 0–45 day lead time when the promise is known, as in Phase 2. |
| `regression.raw_days_from_promise` | R2B before that bounding |
| `regression.late_by_days`, `regression.early_by_days` | The prediction split into lateness and earliness |
| `model_version` | Exact fitted model that produced the score |

### Score several orders

Each row carries its own `features` and an optional `order_id` (up to 128 characters).

```bash
curl -X POST https://team8-delivery-api.onrender.com/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"rows": [
        {"order_id": "example-1", "features": { ...24 features... }},
        {"order_id": "example-2", "features": { ...24 features... }},
        {"order_id": "bad-row",   "features": { ...24 features..., "customer_state": "XX" }}
      ]}'
```

Results come back in the same order, each with its `index` and `order_id`
(predictions trimmed here to the two headline fields; each row's full body matches `/predict`):

```json
{
  "results": [
    {
      "index": 0,
      "order_id": "example-1",
      "prediction": {
        "classification": {
          "late_probability": 0.05184186498640409,
          "…": "…"
        },
        "regression": {
          "predicted_days_from_promise": -9.88466003218192,
          "…": "…"
        }
      }
    },
    {
      "index": 1,
      "order_id": "example-2",
      "prediction": {
        "classification": {
          "late_probability": 0.12239671108893478,
          "…": "…"
        },
        "regression": {
          "predicted_days_from_promise": -0.028580405005691034,
          "…": "…"
        }
      }
    },
    {
      "index": 2,
      "order_id": "bad-row",
      "error": "customer_state must exactly match one of: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO; null is not supported",
      "status_code": 422
    }
  ],
  "successful_rows": 2
}
```

A ready-made file of 10 real held-out orders,
[`examples/batch_example.csv`](examples/batch_example.csv), uses the same columns
as the app's template: upload it on the app's **Batch API demo** page to score
all 10 in one call.

An invalid row does not fail the batch: it gets its own `error` and
`status_code: 422`, and every other row is still scored.

### Python

```python
import json
import requests

BASE = "https://team8-delivery-api.onrender.com"
features = json.load(open("docs/synthetic_prediction.json"))["features"]

requests.get(BASE + "/health", timeout=90).raise_for_status()   # wakes the service if asleep

response = requests.post(BASE + "/predict", json={"features": features}, timeout=60)
response.raise_for_status()
result = response.json()
print(result["classification"]["late_probability"])  # compare with threshold from /model-info
print(result["regression"]["predicted_days_from_promise"])

batch = requests.post(BASE + "/predict/batch", timeout=60, json={"rows": [
    {"order_id": "a", "features": features},
    {"order_id": "b", "features": {**features, "promised_lead_days": 8}},
]}).json()
for row in batch["results"]:
    print(row["order_id"], row.get("prediction", {}).get("classification", {}).get("late_probability"), row.get("error"))
```

### Monitoring

The example below is illustrative (values from the previous deployment).

```bash
curl https://team8-delivery-api.onrender.com/logs/summary
```

```json
{
  "request_count": 3,
  "prediction_count": 3,
  "window_size": 3,
  "window_capacity": 1000,
  "mean_score": 0.11840395776232056,
  "recent_warnings": [
    {
      "timestamp": "2026-10-25T06:31:41.344711+00:00",
      "features": [
        "promised_lead_days"
      ]
    }
  ],
  "counting_policy": "Counts direct prediction HTTP calls and successful classification rows.",
  "storage": "Process-local bounded memory; no order identifiers or feature values retained."
}
```

Counters are kept in memory and reset whenever the Render service restarts. No
order identifiers or feature values are stored.

## Errors

| Status | When | Body |
|---|---|---|
| 422 | Missing, unknown or invalid feature; wrong type; batch of 0 or more than 1,000 rows | `{"detail": "<what is wrong>"}` |
| 503 | A model failed to load at startup | `{"detail": "Classification model is not loaded"}` |
| 500 | Unexpected inference failure (details are logged, not returned) | `{"detail": "Prediction failed; please try again or contact the service owner"}` |

Example: a category the model was not trained on.

```bash
curl -X POST https://team8-delivery-api.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": { ...24 features..., "customer_state": "XX" }}'
```

```json
{
  "detail": "customer_state must exactly match one of: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO; null is not supported"
}
```

## Verify a deployment

```bash
python -m deployment.scripts.verify_http https://team8-delivery-api.onrender.com
```

The script checks health, single and batch equivalence, each model route
against the integrated route, 422 handling and the Swagger page.
