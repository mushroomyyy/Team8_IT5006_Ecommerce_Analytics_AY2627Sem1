# Repository structure

The project is split by lifecycle stage. Existing Phase 2 paths are retained
because the notebooks and tests use relative imports and data resolution.

```text
project-root/
├── Olist_CSV/                 # local source data
├── phase1_eda_dashboard/      # exploratory dashboard
├── phase2/
│   ├── *.ipynb                # final modelling notebooks
│   ├── src/                   # reusable modelling and evaluation code
│   ├── results/               # report tables, figures and metadata
│   ├── test_*.py              # unit tests
│   └── requirements.txt       # modelling/notebook environment
├── deployment/
│   ├── api.py                 # FastAPI and Swagger/OpenAPI endpoints
│   ├── model_service.py       # artifact loading and prediction
│   ├── train_regression.py    # reproducible R1 packaging script
│   ├── streamlit_app.py       # operations prediction UI
│   ├── artifacts/             # packaged model and metadata
│   └── README.md              # local and Render instructions
├── streamlit_release/         # hosted Streamlit entry point
├── docs/                      # project/deployment documentation
├── requirements.txt           # complete root development install
└── README.md                 # project overview and quick start
```

`deployment/` is the production-facing slice. It packages the selected R1
regression model with preprocessing, exposes `/predict` through FastAPI, and
presents the workflow through Streamlit. Classification C3 can be added to the
same service later without changing the regression endpoint contract.
