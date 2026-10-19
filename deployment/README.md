# Regression deployment

This folder contains the first deployment slice for Phase 2: the selected
all-feature OLS regression model (`R1`). Classification is intentionally not
wired in yet.

## Build the model artifact

From the repository root:

```bash
python -m deployment.train_regression
```

This fits the same R1 pipeline used by `phase2/03_regression.ipynb`, using the
Phase 2 history available before 2 June 2018. The fitted preprocessing and
model are stored together in `deployment/artifacts/`.

## Run the API

```bash
uvicorn deployment.api:app --reload
```

Open `/docs` for Swagger UI or `/redoc` for ReDoc. The API exposes `/health`,
`/model-info`, and `POST /predict`. The prediction response includes an
operational late flag, late/early days, and a `follow_up` or `monitor`
recommendation derived from the regression prediction. This is not the C3
classification probability model; it is the regression-only late flag planned
for the current deployment slice.

## Run Streamlit

```bash
streamlit run deployment/streamlit_app.py
```

The app supports historical-order samples and manual feature entry. The Model
Stats tab displays held-out Validation/June metrics and charts from Phase 2;
it does not mix those historical results with live app predictions.

## Render

Use the repository root as the Render service root. A web service can use:

```text
Build command: pip install -r deployment/requirements.txt && python -m deployment.train_regression
Start command: uvicorn deployment.api:app --host 0.0.0.0 --port $PORT
```

For the Streamlit service, use:

```text
Build command: pip install -r deployment/requirements.txt && python -m deployment.train_regression
Start command: streamlit run deployment/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

Training during build keeps the binary artifact reproducible and avoids
committing a large generated model file to Git. If build time is too long for
the chosen Render plan, commit the generated joblib artifact and remove the
training command from the build step.
