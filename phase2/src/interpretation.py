"""Model interpretation: statsmodels coefficient tables for the linear models and RF explanations."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import partial_dependence, permutation_importance

from . import RANDOM_STATE
from .linear_transforms import encoded_scales, readable_names, unit_columns

# Expected sign of each effect on lateness (+1 later or more late, -1 earlier). These are
# working assumptions for the six feature-engineering hypotheses; pass your own to override.
# Keys are raw features (the linear term is used) or exact encoded column names.
DEFAULT_HYPOTHESES = {
    'time': {'is_weekend_approval': 1, 'is_black_friday_period': 1, 'is_december_peak': 1},
    'workload': {'orders_approved_prev_7d': 1, 'approval_lag_hours': 1},
    'order': {'order_item_count': 1, 'order_seller_count': 1, 'order_product_weight_g_sum': 1,
              'order_product_volume_cm3_sum': 1},
    'distance': {'seller_dist_km_mean': 1, 'seller_state_count': 1},
    'payment': {'payment_installments_max': 1, 'payment_value_sum': 1},
    'promise': {'promised_lead_days': -1},
}


def fit_statsmodels(task, design, y, robust=True):
    """OLS (HC3 standard errors when `robust`) or unweighted Logit on the encoded design + intercept."""
    exog = sm.add_constant(design, prepend=True, has_constant='add')
    if task == 'regression':
        return sm.OLS(np.asarray(y, dtype=float), exog).fit(cov_type='HC3' if robust else 'nonrobust')
    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sm.Logit(y, exog).fit(disp=0, maxiter=200)
        if not result.mle_retvals.get('converged', True):  # Quasi-separation: retry with BFGS
            result = sm.Logit(y, exog).fit(disp=0, maxiter=5000, method='bfgs', gtol=1e-8)
    return result


def convergence_report(result, se_limit=10.0):
    """Did the fit converge, and which terms look quasi-separated (huge standard errors)?"""
    converged = bool(getattr(result, 'mle_retvals', {}).get('converged', True))
    se = pd.Series(np.asarray(result.bse, dtype=float), index=result.params.index)
    unstable = se[se > se_limit]
    return {'converged': converged, 'unstable_terms': unstable.round(1).to_dict()}


def assert_coefficients_match(sklearn_pipeline, sm_result, tol=1e-3, exclude_terms=(), max_se=None):
    """Raise unless scikit-learn and statsmodels agree on intercept and slopes; returns the max gap.

    `exclude_terms` and `max_se` (skip terms with a larger statsmodels standard error)
    drop weakly identified terms, such as dummies for a rare level with no late
    orders, whose unpenalised estimate is not unique; list what you skipped in the notebook.
    """
    estimator = sklearn_pipeline[-1]
    slopes = np.ravel(estimator.coef_)
    intercept = float(np.ravel(estimator.intercept_)[0])
    params = sm_result.params
    names = list(params.index)
    values = np.asarray(params, dtype=float)
    if len(values) != len(slopes) + 1:
        raise AssertionError(f'{len(values) - 1} statsmodels slopes vs {len(slopes)} scikit-learn slopes')
    if names[0] != 'const':
        raise AssertionError('statsmodels intercept must come first (use fit_statsmodels)')
    # Align by column name, so the order of the two designs cannot matter
    sklearn_names = ['const'] + readable_names(sklearn_pipeline.named_steps['preprocessor'])
    if sorted(sklearn_names) != sorted(names):
        raise AssertionError('The two designs have different columns: '
                             f'{sorted(set(sklearn_names) ^ set(names))[:5]}')
    sklearn_values = pd.Series(np.r_[intercept, slopes], index=sklearn_names)[names].to_numpy()
    keep = np.array([name not in set(exclude_terms) for name in names])
    if max_se is not None:
        keep &= np.asarray(sm_result.bse, dtype=float) <= max_se
    gaps = np.abs(values - sklearn_values)[keep]
    if not np.allclose(values[keep], sklearn_values[keep], rtol=tol, atol=tol):
        worst = pd.Series(np.abs(values - sklearn_values), index=names)[keep].nlargest(3).round(4)
        raise AssertionError(f'statsmodels and scikit-learn coefficients differ by up to '
                             f'{gaps.max():.2e}: {worst.to_dict()}')
    return float(gaps.max())


def coefficient_table(result, task, preprocessor, cat_cols=(), alpha=0.05):
    """Coefficients with 95% CIs, on the standardised design and on raw scales.

    `coef`, `ci_low`, `ci_high` are on the model scale (log-odds or days) for a 1-SD
    change in a numeric column, or for a category versus its reference. The `effect_*`
    columns convert them: odds ratios for classification, days for regression. Per-unit
    effects use the raw scale (for log1p inputs: per doubling; for dummies: versus the
    reference). Sine/cosine terms have no per-unit reading (NaN).
    """
    scales = encoded_scales(preprocessor)
    unit_of = {term: unit for unit, terms in unit_columns(preprocessor, list(cat_cols)).items()
               for term in terms}
    interval = result.conf_int(alpha=alpha)
    interval = pd.DataFrame(np.asarray(interval), index=result.params.index, columns=['low', 'high'])
    convert = (lambda v: np.exp(np.clip(v, -700, 700))) if task == 'classification' else (lambda value: value)
    effect_type = 'odds ratio' if task == 'classification' else 'days'
    rows = []
    for term in result.params.index:
        coef, low, high = (float(result.params[term]), interval.loc[term, 'low'],
                           interval.loc[term, 'high'])
        row = {'term': term, 'unit': unit_of.get(term, '(intercept)'), 'coef': coef,
               'se': float(result.bse[term]), 'statistic': float(result.tvalues[term]),
               'p_value': float(result.pvalues[term]), 'ci_low': low, 'ci_high': high,
               'effect_type': effect_type}
        if term == 'const':
            kind, basis = 'intercept', 'n/a'
        elif term not in scales.index:
            kind, basis = 'categorical', 'vs reference level'
        else:
            kind = ('log' if term.startswith('log_')
                    else 'cyclic' if term.endswith(('_sin', '_cos')) else 'numeric')
            basis = {'log': 'per doubling', 'numeric': 'per 1 unit'}.get(kind, 'n/a')
        row.update(kind=kind, per_unit_basis=basis)
        if kind in ('numeric', 'log'):
            sd = float(scales[term])
            row['sd'] = sd
            for suffix, value in [('', coef), ('_low', low), ('_high', high)]:
                row[f'effect_per_sd{suffix}'] = convert(value)
                step = np.log(2) / sd if kind == 'log' else 1 / sd
                row[f'effect_per_unit{suffix}'] = convert(value * step)
        elif kind == 'categorical':
            for suffix, value in [('', coef), ('_low', low), ('_high', high)]:
                row[f'effect_per_unit{suffix}'] = convert(value)
        rows.append(row)
    table = pd.DataFrame(rows)
    columns = ['term', 'unit', 'kind', 'coef', 'se', 'statistic', 'p_value', 'ci_low', 'ci_high',
               'effect_type', 'effect_per_sd', 'effect_per_sd_low', 'effect_per_sd_high',
               'per_unit_basis', 'effect_per_unit', 'effect_per_unit_low', 'effect_per_unit_high', 'sd']
    return table.reindex(columns=columns)


def hypothesis_check_table(coef_tables, hypotheses=None, alpha=0.05):
    """One row per hypothesis feature and model: expected sign, observed sign, p-value, verdict.

    `coef_tables` maps a model label (e.g. 'C2') to its `coefficient_table`. Verdict is
    'agree' or 'disagree' when the coefficient is significant at `alpha`, else
    'not significant'; 'not in model' when the feature was not selected.
    """
    hypotheses = hypotheses or DEFAULT_HYPOTHESES
    records = []
    for label, table in coef_tables.items():
        table = table.set_index('term')
        for hypothesis, features in hypotheses.items():
            for feature, expected in features.items():
                matches = [t for t in table.index
                           if t == feature or (table.loc[t, 'unit'] == feature
                                               and t in (feature, f'log_{feature}'))]
                record = {'hypothesis': hypothesis, 'feature': feature, 'model': label,
                          'expected_sign': '+' if expected > 0 else '-'}
                if not matches:
                    records.append({**record, 'term': None, 'coef': np.nan, 'p_value': np.nan,
                                    'observed_sign': '', 'verdict': 'not in model'})
                    continue
                term = matches[0]
                coef, p_value = table.loc[term, 'coef'], table.loc[term, 'p_value']
                observed = 1 if coef > 0 else -1
                verdict = ('not significant' if p_value >= alpha
                           else 'agree' if observed == expected else 'disagree')
                records.append({**record, 'term': term, 'coef': coef, 'p_value': p_value,
                                'observed_sign': '+' if observed > 0 else '-', 'verdict': verdict})
    return pd.DataFrame(records)


def rf_permutation_importance(model, X, y, scoring, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1):
    """Permutation importance on raw features, as the drop in `scoring`, sorted descending."""
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=n_jobs)
    table = pd.DataFrame({'feature': list(X.columns), 'importance_mean': result.importances_mean,
                          'importance_sd': result.importances_std})
    table = table.sort_values('importance_mean', ascending=False, ignore_index=True)
    table['rank'] = np.arange(1, len(table) + 1)
    return table


def rf_partial_dependence(model, X, features, categorical=(), grid_resolution=20, max_rows=5000,
                          random_state=RANDOM_STATE):
    """Average partial dependence per raw feature; returns {feature: DataFrame(value, average)}.

    Classification uses the predicted probability of the late class. Rows are
    subsampled to `max_rows` for speed.
    """
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)
    tables = {}
    for feature in features:
        is_cat = feature in set(categorical)
        result = partial_dependence(model, X, [feature], grid_resolution=grid_resolution,
                                    kind='average', categorical_features=[feature] if is_cat else None)
        tables[feature] = pd.DataFrame({'value': result['grid_values'][0],
                                        'average': result['average'][0]})
    return tables


def plot_partial_dependence(tables, ylabel, save_path=None):
    """One panel per feature from `rf_partial_dependence`."""
    import matplotlib.pyplot as plt
    from . import report_figures as figs

    with plt.rc_context(figs.STYLE):
        fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.6), squeeze=False)
        for ax, (feature, table) in zip(axes[0], tables.items()):
            if table['value'].dtype == object:
                ax.bar(table['value'].astype(str), table['average'], color=figs.BLUE)
                ax.tick_params(axis='x', rotation=60)
            else:
                ax.plot(table['value'], table['average'], color=figs.BLUE)
            ax.set(xlabel=feature, ylabel=ylabel)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
    return fig


def agreement_table(rf_importance, coef_table, top=10, alpha=0.05):
    """RF top-`top` features beside whether the linear model finds the unit significant."""
    significance = (coef_table[coef_table['unit'].ne('(intercept)')]
                    .groupby('unit')['p_value'].min())
    table = rf_importance.head(top)[['rank', 'feature', 'importance_mean']].copy()
    table['linear_min_p'] = table['feature'].map(significance)
    table['linear_significant'] = (table['linear_min_p'].lt(alpha).astype('boolean')
                                   .mask(table['linear_min_p'].isna()))
    return table.reset_index(drop=True)
