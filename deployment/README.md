# Deployment code moved

The integrated C3 + R1 application is maintained under [`phase3/`](../phase3/README.md).

Run `python -m uvicorn phase3.backend.api:app --port 8002` and
`python -m streamlit run phase3/frontend/app.py`. The `streamlit_release/`
entry point remains the Phase 1 dashboard.
