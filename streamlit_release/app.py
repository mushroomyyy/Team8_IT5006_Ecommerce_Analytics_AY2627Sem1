"""Production entry point for Streamlit Community Cloud."""

from pathlib import Path
import os
import runpy


os.environ["OLIST_HOSTED_RELEASE"] = "1"
shared_app = Path(__file__).resolve().parents[1] / "phase1_eda_dashboard" / "app.py"
runpy.run_path(shared_app, run_name="__main__")
