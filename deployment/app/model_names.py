"""Short labels for the model codes; mirrors src/model_labels.py because the app never imports src."""

MODEL_LABELS = {
    'C0': 'No-skill baseline', 'C1': 'Logistic (all)', 'C2': 'Logistic (CV-stepwise)', 'C2B': 'Logistic (BIC)',
    'C3': 'Random Forest (tuned)', 'C3d': 'Random Forest (default)', 'C4': 'Decision tree (tuned)',
    'R0a': 'Median baseline', 'R0b': 'Olist promise', 'R1': 'OLS (all)', 'R2': 'OLS (CV-stepwise)', 'R2B': 'OLS (BIC)',
    'R3': 'Random Forest (tuned)', 'R3d': 'Random Forest (default)', 'R4': 'Decision tree (tuned)',
}


def short_label(code):
    """Readable name for a model code; unknown codes are shown as they are."""
    return MODEL_LABELS.get(code, code)
