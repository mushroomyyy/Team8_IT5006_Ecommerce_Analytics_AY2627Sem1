# Connect a frontend to the model API

Teammates do **not** need Render accounts or workspace access to use this public
API. Render workspace access is only needed to manage hosting, configuration,
logs or deployments. No API key is required by this demonstration service.

- Base URL: https://team8-delivery-api.onrender.com
- Interactive schema: https://team8-delivery-api.onrender.com/docs
- Health: https://team8-delivery-api.onrender.com/health
- Input feature names/categories: `GET /model-info`
- One order: `POST /predict` with `{"features": {all 24 engineered inputs}}`
- Several orders: `POST /predict/batch` with
  `{"rows": [{"order_id": "synthetic-1", "features": {all 24 inputs}}]}`

## Test the connection

From the repository root, this request uses an invented order without real
transaction data:

```bash
curl --fail https://team8-delivery-api.onrender.com/health
curl --fail https://team8-delivery-api.onrender.com/predict \
  -H 'Content-Type: application/json' \
  --data-binary @phase3/docs/synthetic_prediction.json
```

The response contains `classification.late_probability` and
`regression.predicted_days_from_promise`. Positive regression days mean later than
the promise; negative means earlier. The regression target uses a 45-day delivery
cap. Risk is a ranking score, not a guarantee or a fixed classification decision.
Invalid requests return 422; batch row errors retain their input index/ID. Handle
connection failures, non-200 responses and timeouts explicitly. Allow 60 seconds for
requests because a sleeping free Render service may need time to wake.

## Run our Streamlit application

```bash
DELIVERY_API_URL=https://team8-delivery-api.onrender.com \
  python -m streamlit run phase3/frontend/app.py
```

For Streamlit Community Cloud, use branch `classification-deployment`, main file
`phase3/frontend/app.py`, Python 3.13 and this root-level secret:

```toml
DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
```

Streamlit handles historical input files, cutoff eligibility, daily risk ranking,
follow-up selection and retrospective outcomes in its Python server. It sends
**every eligible order** to `/predict/batch` before selecting the top X% for human
follow-up. Render only validates inputs and runs fitted models; no historical demo endpoints
are needed. Streamlit never loads fitted models or preprocessing.

To publish Community Cloud from this repository, the publisher needs GitHub
repository admin permissions. If Andrew only has write access, Jordan can publish
with these settings. Public frontend deployment remains pending at Andrew’s request.
