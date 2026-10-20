"""Operations proof of concept; all model operations use the integrated HTTP API."""
from __future__ import annotations

import calendar
from datetime import date, timedelta
import io
import json
import math
import os
from pathlib import Path
import sys

# Streamlit runs this file directly in Cloud, Docker and AppTest.
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from phase3.frontend.orchestration import (api_request, cached_worklist, load_demo_orders,
    load_demo_samples, model_versions, reveal_outcomes, snapshot_revision)
from phase3.frontend.simulation import configuration, SimulationDataError

import pandas as pd
import requests
import streamlit as st

LABELS = {
    'order_approved_day_of_week': ('Approval weekday', 'Monday=1 through Sunday=7.'),
    'order_approved_day_of_month': ('Approval day of month', 'Day on which payment approval occurred.'),
    'order_approved_month': ('Approval month', 'Calendar month of approval.'),
    'order_item_count': ('Number of items', 'Number of product line items in the order.'),
    'order_seller_count': ('Number of sellers', 'Distinct sellers fulfilling this order.'),
    'order_frieght_value_sum': ('Freight value (R$)', 'Sum of freight charged across order items.'),
    'order_product_weight_g_sum': ('Total product weight (g)', 'Sum of catalogue product weights.'),
    'order_product_category_count': ('Product categories', 'Distinct product categories in this order.'),
    'seller_dist_km_mean': ('Mean seller distance (km)', 'Mean great-circle seller-to-customer distance across distinct sellers.'),
    'seller_state_count': ('Seller states', 'Distinct states represented by order sellers.'),
    'payment_value_sum': ('Payment value (R$)', 'Total value across all order payments.'),
    'promised_lead_days': ('Promised lead time (days)', 'Promised delivery date minus approval date; negative values are allowed.'),
    'approval_lag_hours': ('Purchase-to-approval time (hours)', 'Elapsed hours from purchase to payment approval.'),
    'order_product_volume_cm3_sum': ('Total product volume (cm³)', 'Sum of catalogue length × height × width per item.'),
    'order_product_photos_mean': ('Mean catalogue photos', 'Mean photo count across order items.'),
    'freight_to_price_ratio': ('Freight-to-price ratio', 'Freight total divided by product price total.'),
    'payment_installments_max': ('Maximum instalments', 'Largest instalment count among order payments.'),
    'is_weekend_approval': ('Weekend approval', 'Derived from weekday: Saturday or Sunday.'),
    'is_black_friday_period': ('Black Friday period', 'Approval during the fixed 20–30 November 2017 feature window.'),
    'is_december_peak': ('December peak', 'Derived from approval month: December.'),
    'orders_approved_prev_7d': ('Platform approvals in previous 7 days', 'All platform approvals in seven completed days, excluding approval day.'),
    'customer_state': ('Customer state', 'Brazilian destination state.'),
    'route_type': ('Shipping route', 'Whether all, none, or some sellers are interstate.'),
    'payment_combination': ('Payment methods', 'Sorted unique payment methods joined with +.'),
}
GROUPS = {
    'Order and timing': ['order_approved_day_of_week','order_approved_month','order_approved_day_of_month','promised_lead_days','approval_lag_hours','orders_approved_prev_7d','is_weekend_approval','is_december_peak','is_black_friday_period'],
    'Items and shipping': ['order_item_count','order_seller_count','order_product_category_count','seller_state_count','order_frieght_value_sum','order_product_weight_g_sum','order_product_volume_cm3_sum','order_product_photos_mean','seller_dist_km_mean','freight_to_price_ratio'],
    'Payment': ['payment_value_sum','payment_installments_max','payment_combination'],
    'Customer and route': ['customer_state','route_type'],
}
COUNTS = {'order_item_count','order_seller_count','order_product_category_count','seller_state_count','payment_installments_max','orders_approved_prev_7d'}
INTEGERS = COUNTS | {'order_approved_day_of_week','order_approved_month','order_approved_day_of_month','promised_lead_days'}
BINARY = {'is_weekend_approval','is_december_peak','is_black_friday_period'}
# Columns this app adds to downloaded CSVs; dropped when such a file is uploaded again.
UPLOAD_OUTPUT_COLUMNS = {'late_probability','is_predicted_late','threshold','threshold_policy','delivery_status','recommended_action',
                         'above_watch_list_boundary','watch_list_boundary','drift_warnings','model_code','model_version',
                         'error','risk_rank','rank','watch_list','actual_late','predicted_days_from_promise',
                         'raw_days_from_promise','late_by_days','early_by_days','clipping_applied','clipping_policy','target','score_interpretation'}


@st.cache_data(ttl=0, show_spinner=False)
def model_info(api_url):
    return api_request(api_url, '/model-info')


def validate_form(features, stats):
    """Immediate form feedback; the API remains the authoritative validator."""
    errors = []
    expected = stats['features']['numeric'] + stats['features']['categorical']
    if set(features) != set(expected):
        return ['Supply exactly the 24 engineered input features.']
    if all(features[name] is None for name in stats['features']['numeric']):
        errors.append('Supply at least one numeric feature; all numeric inputs cannot be missing.')
    for name in stats['features']['categorical']:
        if features[name] not in stats['categories'][name]:
            errors.append(f'{name}: choose a trained category: {stats["categories"][name]}')
    for name in stats['features']['numeric']:
        value = features[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or abs(value) > 1e12 or not math.isfinite(value):
            errors.append(f'{name}: enter a finite number with magnitude at most 1e12.'); continue
        if name in INTEGERS | BINARY and int(value) != value:
            errors.append(f'{name}: enter an integer.')
        if name in COUNTS and value < (0 if name == 'orders_approved_prev_7d' else 1):
            errors.append(f'{name}: count is below its minimum.')
        if name not in INTEGERS | BINARY and value < 0:
            errors.append(f'{name}: enter a nonnegative number.')
        if name in BINARY and value not in (0, 1):
            errors.append(f'{name}: expected 0 or 1.')
    if errors:
        return errors
    month, day, weekday = (features.get(k) for k in ['order_approved_month','order_approved_day_of_month','order_approved_day_of_week'])
    if month is not None and (not isinstance(month, (int,float)) or month not in range(1,13)):
        errors.append('Approval month must be 1–12.')
    elif month is not None and day is not None and (day < 1 or day > calendar.monthrange(2000,int(month))[1]):
        errors.append('Invalid approval day/month combination.')
    if day is not None and not 1 <= day <= 31:
        errors.append('Approval day must be 1–31.')
    if weekday is not None and weekday not in range(1,8):
        errors.append('Approval weekday must be 1–7.')
    if weekday is not None and features.get('is_weekend_approval') is not None and features['is_weekend_approval'] != int(weekday >= 6):
        errors.append('Weekend flag must match approval weekday.')
    if month is not None and features.get('is_december_peak') is not None and features['is_december_peak'] != int(month == 12):
        errors.append('December flag must match approval month.')
    return errors


def range_warnings(features, stats):
    for name, limits in stats.get('numeric_ranges', {}).items():
        value = features.get(name)
        if value is None: continue
        label = LABELS.get(name, (name,''))[0]
        if value < limits['min'] or value > limits['max']:
            st.warning(f'{label}: outside refit range {limits["min"]:g}–{limits["max"]:g}. This input is extrapolating beyond training history.')
        elif value < limits['p01'] or value > limits['p99']:
            st.warning(f'{label}: outside the refit 1st–99th percentile ({limits["p01"]:g}–{limits["p99"]:g}). Review this unusual input.')


def structured_inputs(sample, stats, prefix):
    features = {}
    for title, names in GROUPS.items():
        with st.container(border=True):
            st.subheader(title)
            for name in names:
                label, help_text = LABELS[name]
                help_text += f' Feature: {name}'
                value = sample.get(name)
                key = f'{prefix}_{name}'
                if name in stats['features']['categorical']:
                    options = stats['categories'][name]
                    features[name] = st.selectbox(label, options, index=options.index(value) if value in options else 0, key=key, help=help_text)
                    continue
                cols = st.columns([4,1])
                missing = cols[1].checkbox('Missing', value=value is None, key=key+'_missing', help='Numeric null uses the median fitted on training history.')
                default = stats.get('numeric_ranges',{}).get(name,{}).get('median',0) if value is None else value
                if name in BINARY:
                    derived = (int(features['order_approved_day_of_week'] >= 6) if name == 'is_weekend_approval' and features.get('order_approved_day_of_week') is not None else int(features['order_approved_month'] == 12) if name == 'is_december_peak' and features.get('order_approved_month') is not None else int(default))
                    checked = cols[0].checkbox(label, value=bool(derived), key=key+(f'_{derived}' if name != 'is_black_friday_period' else ''), disabled=missing or name != 'is_black_friday_period', help=help_text)
                    # Derived flags follow the current date fields on every rerun.
                    features[name] = None if missing else (int(checked) if name == 'is_black_friday_period' else derived)
                elif name == 'order_approved_day_of_week':
                    selected = cols[0].selectbox(label, list(range(1,8)), index=max(0,min(6,int(default)-1)), format_func=lambda n: f'{n} · {calendar.day_name[n-1]}', key=key, disabled=missing, help=help_text)
                    features[name] = None if missing else int(selected)
                else:
                    integer = name in INTEGERS
                    minimum = (0 if name == 'orders_approved_prev_7d' else 1) if name in COUNTS else (1 if name in {'order_approved_month','order_approved_day_of_month'} else None if name == 'promised_lead_days' else 0.0)
                    kwargs = {'step':1, 'value':int(default), 'min_value':int(minimum) if minimum is not None else None} if integer else {'step':0.1, 'value':float(default), 'min_value':float(minimum)}
                    current = cols[0].number_input(label, key=key, disabled=missing, help=help_text, **kwargs)
                    features[name] = None if missing else (int(current) if integer else float(current))
    return features


def prepare_upload(incoming,feature_cols):
    """Return (kept columns, batch rows, error) for an uploaded engineered-feature CSV."""
    if len(incoming)>1000:
        return incoming,[],'Upload at most 1,000 orders.'
    missing=[c for c in feature_cols if c not in incoming]
    unexpected=[c for c in incoming if c not in feature_cols and c!='order_id' and c not in UPLOAD_OUTPUT_COLUMNS]
    if missing or unexpected:
        return incoming,[],'; '.join(filter(None,[f'Missing feature columns: {", ".join(missing)}' if missing else '',
                                                  f'Unrecognised columns: {", ".join(unexpected)}' if unexpected else '']))
    incoming=incoming[(['order_id'] if 'order_id' in incoming else [])+feature_cols].reset_index(drop=True)
    # Plain Python values keep full float precision (to_json rounds to at most 15 digits).
    values=incoming[feature_cols].astype(object)
    features=values.where(values.notna(),None).to_dict(orient='records')
    order_ids=incoming['order_id'].astype(str).tolist() if 'order_id' in incoming else [None]*len(incoming)
    rows=[{'features':f,**({'order_id':i} if i is not None else {})} for f,i in zip(features,order_ids)]
    return incoming,rows,None


def show_error(exc):
    st.error(f'Unable to complete this request: {exc}')
    st.info('Check the API URL and retry. A sleeping Render service may need up to a minute to start.')


def result_panel(result, stats, api_url, features):
    classification, regression = result['classification'], result['regression']
    st.subheader('Prediction results')
    st.caption('Results for the last submitted inputs. Submit again after editing.')
    st.metric('Estimated late-delivery risk', f"{classification['late_probability']:.1%}")
    days = regression['predicted_days_from_promise']
    st.metric('Estimated days from promise', f'{days:+.2f} days')
    st.caption('Positive = later than the promised date; negative = earlier. This is not time from approval.')
    st.caption(regression.get('clipping_policy', 'When the promise is available, predicted approval-to-delivery duration is bounded to 0–45 days, matching Phase 2 evaluation. With a missing promise, the raw estimate is shown.'))
    st.info('C3 risk ranks orders; R1 estimates timing relative to the promise. The models can disagree. A low score does not guarantee on-time delivery.')
    for warning in classification.get('drift_warnings', []):
        st.warning(warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning))


def check_order(api_url, stats, samples):
    st.title('Single-order API demo')
    st.write('These are the same C3 and R1 prediction services used by the daily worklist. Choose an example, review the engineered features, then submit both models in one API call.')
    if not samples:
        st.info('No example orders available.'); return
    if st.session_state.get('open_order'):
        st.session_state['sample_picker'] = st.session_state.pop('open_order')
    sample_id = st.selectbox('Held-out historical order', [r['order_id'] for r in samples], key='sample_picker')
    sample = next(r for r in samples if r['order_id'] == sample_id)
    sample = sample.get('features', sample)
    if st.session_state.get('last_sample') != sample_id:
        for key in list(st.session_state):
            if key.startswith('input_') or key in ('last_result', 'last_features'):
                del st.session_state[key]
        st.session_state['last_sample'] = sample_id
    left, right = st.columns([3, 2])
    with left:
        features = structured_inputs(sample, stats, 'input_' + sample_id)
        with st.expander('Advanced: edit raw JSON'):
            use_raw = st.checkbox('Use raw JSON instead of the form', key='input_use_raw')
            raw = st.text_area('Engineered features (JSON)', json.dumps({k: sample[k] for k in stats['features']['numeric'] + stats['features']['categorical']}, indent=2), height=260, key='input_raw')
        if use_raw:
            try:
                features = json.loads(raw)
                if not isinstance(features, dict): raise ValueError('Features must be a JSON object.')
            except ValueError as exc:
                st.error(str(exc)); features = {}
        errors = validate_form(features, stats)
        for error in errors: st.error(error)
        if not errors: range_warnings(features, stats)
        if st.button('Predict delivery', type='primary', disabled=bool(errors)):
            try:
                with st.spinner('Scoring both models with the API…'):
                    result = api_request(api_url, '/predict', {'features': features})
                st.session_state['last_result'] = result
                st.session_state['last_features'] = features
            except (ValueError, requests.RequestException, SimulationDataError) as exc:
                show_error(exc)
    with right:
        with st.container(border=True):
            if st.session_state.get('last_result'):
                result_panel(st.session_state['last_result'], stats, api_url, st.session_state['last_features'])
            else:
                st.subheader('Ready to predict')
                st.write('Classification risk and regression days appear after submission.')


def navigate_day(delta):
    st.session_state['simulation_day'] += timedelta(days=delta)


def percent(value):
    return 'Not defined' if value is None else f'{value:.1%}'


def daily_page(api_url, metadata):
    st.title('Morning delivery follow-up')
    st.write('Imagine today is a day in June or early July 2018. Which orders approved yesterday should operations follow up on? Each day is a separate batch.')
    config = configuration(load_demo_orders())
    minimum, maximum = date.fromisoformat(config['minimum_date']), date.fromisoformat(config['maximum_date'])
    if 'simulation_day' not in st.session_state:
        st.session_state['simulation_day'] = date.fromisoformat(config['default_date'])
    previous, picker, following = st.columns([1, 3, 1])
    previous.button('Previous day', disabled=st.session_state['simulation_day'] <= minimum, on_click=navigate_day, args=(-1,))
    day = picker.date_input('Simulated current date', min_value=minimum, max_value=maximum, key='simulation_day')
    following.button('Next day', disabled=day >= maximum, on_click=navigate_day, args=(1,))
    policy = st.columns(2)
    review = policy[0].slider('Follow-up proportion (%)', 0, 100, 30, step=5)
    remaining = policy[1].number_input('Minimum days remaining before promise', min_value=0, max_value=365, value=1, step=1)
    st.caption(f'All eligible orders are scored. Select the top {review}% by risk for human follow-up; 30% is the initial demo workload. Exclude promises due within one day by default (eligible promises must be strictly more than the selected minimum days away). These are configurable operational assumptions, not a validated universal policy. Selection rounds up; ties use order ID.')
    query = (api_url, model_versions(metadata), day.isoformat(), review, remaining, snapshot_revision())
    if st.session_state.get('worklist_policy') != query:
        st.session_state['worklist_policy'] = query
        st.session_state.pop('revealed_outcomes', None)
    with st.spinner('Building the worklist as of the simulation cutoff…'):
        result = cached_worklist(api_url, model_versions(metadata), day, review / 100, remaining, snapshot_revision())
    st.subheader(f"Simulated today: {result['simulation_date']}")
    st.write(f"**Scoring cutoff:** {result['cutoff']} · {config['timezone']}  |  **Previous approval batch:** {result['batch_date']}")
    st.caption('Batch: orders approved on the previous calendar day.')
    cols = st.columns(4)
    for column, name in zip(cols, ['total', 'eligible', 'excluded', 'selected']):
        column.metric(name.capitalize() + ' orders', result['counts'][name])
    reasons = result.get('exclusion_reasons', {})
    if reasons:
        st.write('Exclusion reasons (an order can have multiple reasons):')
        st.dataframe(pd.DataFrame([{'Reason': k.replace('_', ' '), 'Orders': v} for k, v in reasons.items()]), hide_index=True, width='stretch')
    orders = result.get('orders', [])
    if orders:
        frame = pd.DataFrame(orders)
        columns = ['rank', 'order_id', 'late_probability', 'recommendation', 'promised_delivery_date', 'predicted_days_from_promise']
        display = frame[[c for c in columns if c in frame]].copy()
        display['promised_delivery_date'] = display['promised_delivery_date'].str[:10]
        st.subheader('Risk-ranked eligible orders')
        st.caption('Follow-up recommended marks the selected review workload. Other orders are not guaranteed on time. Regression reports days relative to promise (positive = late, negative = early), after bounding approval-to-delivery duration to 0–45 days when possible; it does not change ranking.')
        st.dataframe(display, hide_index=True, width='stretch', column_config={
            'rank': st.column_config.NumberColumn('Rank', width='small'),
            'order_id': st.column_config.TextColumn('Order ID', width='medium'),
            'late_probability': st.column_config.NumberColumn('C3 risk', format='percent', width='small'),
            'recommendation': st.column_config.TextColumn('Follow-up', width='medium'),
            'promised_delivery_date': st.column_config.TextColumn('Promised date', width='small'),
            'predicted_days_from_promise': st.column_config.NumberColumn('R1 days ±', format='%.2f', width='small', help='Estimated days from promise: positive late, negative early.')})
        st.download_button('Download prioritised worklist', display.to_csv(index=False), f'worklist_{day}.csv', 'text/csv')
        order_id = st.selectbox('Inspect a ranked order', frame.order_id.tolist())
        selected = next(row for row in orders if row['order_id'] == order_id)
        with st.expander('Selected order details', expanded=False):
            st.json(selected)
        if st.button('Open selected order in API demo'):
            st.session_state['worklist_selected'] = selected
            st.session_state['open_order'] = order_id
            st.session_state['next_page'] = 'Single-order API demo'
            st.rerun()
    else:
        st.info('No eligible orders for this day and policy. Try another day or review the eligibility assumptions.')
    st.subheader('Retrospective evaluation')
    st.caption('Eventual outcomes are hidden by default and were unavailable at this cutoff. Changing the date or policy hides them again.')
    if st.button('Reveal eventual outcomes', key='reveal_outcomes'):
        st.session_state['revealed_outcomes'] = reveal_outcomes(result)
    if st.session_state.get('revealed_outcomes'):
        historical = st.session_state['revealed_outcomes']
        metrics = historical['metrics']
        st.warning('Retrospective outcomes: outside the simulated knowledge at prediction time. These evaluate this day’s filtered policy, not the reported monthly top-30% result.')
        st.dataframe(pd.DataFrame([{'Population': label, **metrics[label]} for label in ['all', 'eligible', 'selected']]), hide_index=True, width='stretch')
        summary = st.columns(3)
        summary[0].metric('Late rate among selected known outcomes', percent(metrics.get('selected_late_rate')))
        summary[1].metric('Coverage of eligible known late orders', percent(metrics.get('eligible_late_coverage')))
        summary[2].metric('Coverage of all known late orders', percent(metrics.get('all_late_coverage')))
        st.caption('Late rate denominator: selected known outcomes. Coverage denominators: eligible known late orders and all known late orders respectively. Unknown outcomes are counted above and excluded from those denominators; undefined rates are not zero.')
        values = historical.get('outcomes', {})
        st.dataframe(pd.DataFrame([{'order_id': k, 'Eventual outcome': {0: 'On time', 1: 'Late'}.get(v, 'Unknown')} for k, v in values.items()]), hide_index=True, width='stretch')
    with st.expander('Simulation assumptions and limits'):
        st.write(f"Model available from {config.get('model_available_date', minimum)}. Dates are restricted to the supported replay period.")
        for limitation in config.get('limitations', []): st.write(limitation)


def scored_upload_frame(incoming, response):
    """Keep original inputs beside API results so a downloaded CSV can be rescored."""
    output = []
    for item in response['results']:
        prediction = item.get('prediction') or {}
        output.append({**prediction.get('classification', {}), **prediction.get('regression', {}), 'error': item.get('error')})
    return pd.concat([incoming, pd.DataFrame(output)], axis=1)


def batch_upload(api_url, stats):
    st.title('Batch API demo')
    st.caption('Upload engineered feature columns with optional order_id. Blank numeric cells become null. At most 1,000 rows. This scores both models without applying a daily follow-up policy.')
    feature_cols = stats['features']['numeric'] + stats['features']['categorical']
    st.download_button('Download feature template', pd.DataFrame(columns=['order_id'] + feature_cols).to_csv(index=False), 'feature_template.csv', 'text/csv')
    upload = st.file_uploader('Engineered order features', type=['csv'])
    if upload is not None and st.button('Score uploaded orders'):
        try:
            incoming = pd.read_csv(io.BytesIO(upload.getvalue()), float_precision='round_trip')
            incoming, rows, error = prepare_upload(incoming, feature_cols)
            if error: st.error(error); return
            response = api_request(api_url, '/predict/batch', {'rows': rows})
            scored = scored_upload_frame(incoming, response)
            st.dataframe(scored, hide_index=True, width='stretch')
            st.download_button('Download scored CSV', scored.to_csv(index=False), 'batch_scores.csv', 'text/csv')
        except (ValueError, requests.RequestException, pd.errors.ParserError) as exc:
            show_error(exc)


def guidance(metadata):
    st.title('Models and guidance')
    st.write('C3 tuned Random Forest estimates late-delivery risk for ranking. R1 OLS estimates days relative to the promised delivery date. Neither output is a delivery guarantee; risk scores are not claimed to be calibrated.')
    st.write('The daily list ranks eligible orders by C3 risk and applies the chosen review proportion. R1 timing provides additional context, including disagreements. Single-order predictions do not assign a threshold-based action.')
    stats = metadata['classification']
    june = stats.get('evaluation', {}).get('June', {})
    if june.get('avg_precision') is not None:
        st.metric('Historical June average precision (C3)', f"{june['avg_precision']:.2%}")
    regression_metrics = metadata.get('regression', {}).get('evaluation', {})
    if regression_metrics.get('june_test_rmse') is not None:
        st.metric('Historical June RMSE (R1)', f"{regression_metrics['june_test_rmse']:.3f} days")
    if regression_metrics.get('june_test_mae') is not None:
        st.metric('Historical June MAE (R1)', f"{regression_metrics['june_test_mae']:.2f} days")
    st.caption('Historical monthly performance does not guarantee this daily filtered policy’s results. Use the retrospective evaluation for explicit day and policy denominators.')
    st.warning('Historical 2018 data and snapshot features may not represent current operations. Numeric nulls use fitted training medians. Historical status timing cannot always be reconstructed; see the simulation limitations.')
    with st.expander('Model metadata and input schema'):
        st.json(metadata)


def main():
    st.set_page_config(page_title='Delivery Watch · C3 + R1', page_icon='📦', layout='wide')
    api_url = os.environ.get('DELIVERY_API_URL', os.environ.get('CLASSIFICATION_API_URL', 'http://127.0.0.1:8002'))
    st.sidebar.title('Delivery Watch')
    st.sidebar.caption('Operations & logistics · Historical simulation')
    if st.sidebar.button('Retry API connection'):
        model_info.clear()
        cached_worklist.clear()
    try:
        if st.session_state.get('api_connected'):
            api_request(api_url, '/health'); metadata = model_info(api_url)
        else:
            with st.spinner('Connecting / waking up the API. A sleeping service can take up to a minute…'):
                api_request(api_url, '/health'); metadata = model_info(api_url)
        service_identity = (api_url, model_versions(metadata))
        if st.session_state.get('prediction_service') != service_identity:
            st.session_state.pop('last_result', None)
            st.session_state.pop('last_features', None)
            st.session_state['prediction_service'] = service_identity
        st.session_state['api_connected'] = True
        st.sidebar.success('API connected · C3 + R1')
    except (ValueError, requests.RequestException, SimulationDataError) as exc:
        st.session_state['api_connected'] = False
        st.sidebar.error('API unavailable'); show_error(exc)
        st.sidebar.caption(api_url); return
    st.sidebar.caption(api_url)
    if st.session_state.get('next_page'):
        st.session_state['workspace'] = st.session_state.pop('next_page')
    page = st.sidebar.radio('Workspace', ['Daily simulation', 'Single-order API demo', 'Batch API demo', 'Models and guidance'], key='workspace')
    try:
        stats = metadata['classification']
        if page == 'Daily simulation': daily_page(api_url, metadata)
        elif page == 'Single-order API demo':
            data = load_demo_samples()
            samples = data.get('samples', []) if isinstance(data, dict) else data
            extra = st.session_state.get('worklist_selected')
            if extra and extra['order_id'] not in [r['order_id'] for r in samples]: samples.append(extra)
            check_order(api_url, stats, samples)
        elif page == 'Batch API demo': batch_upload(api_url, stats)
        else: guidance(metadata)
    except (ValueError, requests.RequestException, SimulationDataError) as exc:
        show_error(exc)
    st.caption('C3 + R1 · HTTP API · Historical simulation')


if __name__ == '__main__': main()
