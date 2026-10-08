"""Model preprocessing shared by the classification and regression pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(num_cols, cat_cols, scale=False, reference_categories=False):
    """Return an unfitted ColumnTransformer: median imputation and one-hot encoding.

    `scale=True` standardises the numeric columns after imputation (linear models).
    `reference_categories=True` drops the modal development-training state and
    route categories, for an intercept-compatible design. The legacy default
    retains full one-hot state and route encoding.
    Use it as the first step of a Pipeline so it is fitted on training rows only.
    """
    numeric = SimpleImputer(strategy='median')
    if scale:
        numeric = Pipeline([('imputer', numeric), ('scaler', StandardScaler())])
    ordinary_cats = [column for column in cat_cols if column != 'payment_combination']
    transformers = [('numeric', numeric, num_cols)]
    if ordinary_cats:
        references = {'customer_state': 'SP', 'route_type': '1. All interstate'}
        if reference_categories:
            unsupported = set(ordinary_cats) - references.keys()
            if unsupported:
                raise ValueError(f'No reference category specified for {sorted(unsupported)}')
            encoder = OneHotEncoder(drop=[references[column] for column in ordinary_cats],
                                    handle_unknown='ignore')
        else:
            encoder = OneHotEncoder(handle_unknown='ignore')
        transformers.append(('categorical', encoder, ordinary_cats))
    if 'payment_combination' in cat_cols:
        # The single-method credit-card group is the explicit reference.
        # Categories are learned only when the training-fold pipeline is fitted.
        transformers.append(('payment_combination',
                             OneHotEncoder(drop=['credit_card'], handle_unknown='ignore'),
                             ['payment_combination']))
    return ColumnTransformer(transformers, sparse_threshold=0)
