"""Production entry point for the model deployment UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.streamlit_app import *  # noqa: F401,F403,E402
