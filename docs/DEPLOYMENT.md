# Deployment checklist

## Local

From the repository root and an activated virtual environment:

```bash
python -m deployment.train_regression
python -m streamlit run deployment/streamlit_app.py --server.port 8503
python -m uvicorn deployment.api:app --reload --port 8000
```

The API provides `/health`, `/model-info`, `/predict`, `/docs` (Swagger UI),
and `/redoc` (ReDoc). The prediction response includes the regression result
and the operational follow-up flag.

## Render

Web service build command:

```text
pip install -r deployment/requirements.txt && python -m deployment.train_regression
```

Web service start command:

```text
python -m uvicorn deployment.api:app --host 0.0.0.0 --port $PORT
```

Streamlit service start command:

```text
python -m streamlit run deployment/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

The artifact must be trained and served with the pinned scikit-learn version
in `deployment/requirements.txt` because serialized preprocessing pipelines
are version-sensitive.
