"""Model preprocessing shared by the classification and regression pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(num_cols, cat_cols, scale=False):
    """Return an unfitted ColumnTransformer: median imputation and one-hot encoding.

    `scale=True` standardises the numeric columns after imputation (linear models).
    Use it as the first step of a Pipeline so it is fitted on training rows only.
    """
    numeric = SimpleImputer(strategy='median')
    if scale:
        numeric = Pipeline([('imputer', numeric), ('scaler', StandardScaler())])
    return ColumnTransformer([
        ('numeric', numeric, num_cols),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], sparse_threshold=0)  # Keep a dense matrix for the tree models
