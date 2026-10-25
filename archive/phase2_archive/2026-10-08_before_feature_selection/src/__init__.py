"""Shared Phase 2 helpers, extracted from model_dev_v2.ipynb so every notebook reuses one copy."""

RANDOM_STATE = 42
LOOKBACK = 45   # Days between the newest training approval and the inference date
PXD = 365       # Length of the train-test window in days
