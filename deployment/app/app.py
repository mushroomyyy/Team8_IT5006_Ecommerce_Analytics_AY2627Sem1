"""Operations proof of concept; all model operations use the integrated HTTP API."""
from __future__ import annotations

from bisect import bisect_left
import calendar
from datetime import date, timedelta
import io
import json
import math
import os
from pathlib import Path
import sys
import time

# Streamlit runs this file directly in Cloud, Docker and AppTest.
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Streamlit Cloud reruns this file after a push without restarting Python, so sibling
# modules imported by an earlier version can be stale; reload any whose source changed.
import importlib
import deployment.app.simulation as _simulation_module
import deployment.app.orchestration as _orchestration_module
import deployment.app.explore as _explore_module
for _module in (_simulation_module, _orchestration_module, _explore_module):
    _mtime = os.path.getmtime(_module.__file__)
    if getattr(_module, '_source_mtime', None) != _mtime:
        _module = importlib.reload(_module)
        _module._source_mtime = _mtime
from deployment.app.orchestration import (HEALTH_TIMEOUT, WAKE_BUDGET_SECONDS, api_request, build_day_worklist,
    load_demo_orders, load_demo_samples, load_month_inputs, model_versions, reveal_outcomes, snapshot_revision,
    wait_for_service)
from deployment.app.explore import explore_page
from deployment.app.model_names import short_label
from deployment.app.simulation import (REPLAY_MONTHS, buffer_ranking_scores, configuration, month_replay, month_summary, monthly_ranking,
    SimulationDataError, timing_accuracy)

import altair as alt
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

PAGES = ['Start here', 'Daily simulation', 'Month-end review', 'Explore the data', 'Single-order API demo', 'Batch API demo', 'Models and guidance']
PAGE_ICONS = {'Start here': '🏁', 'Daily simulation': '📋', 'Month-end review': '📈', 'Explore the data': '🗺️',
              'Single-order API demo': '🔍', 'Batch API demo': '📤', 'Models and guidance': 'ℹ️'}
API_PAGES = {'Daily simulation', 'Single-order API demo', 'Batch API demo', 'Models and guidance'}
CONNECTION_RECHECK_SECONDS = 300
# One palette across pages: orange marks the follow-up workload, slate everything else.
FOLLOW, OTHER, INK, MUTED, TEAL = '#E67E22', '#9FB3C8', '#16324F', '#5B7083', '#2A9D8F'
FOLLOW_LABEL, OTHER_LABEL = 'Follow up', 'No follow-up'
STYLE = """<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.dw-lede {color: #5B7083; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1rem;}
div[data-testid="stMetric"] {background: #FFFFFF;}
/* Sidebar navigation: each page is a full-width, clearly clickable card. */
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 6px; width: 100%; flex-direction: column; align-items: stretch;}
section[data-testid="stSidebar"] div[role="radiogroup"] > div {width: 100%;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  width: 100%; box-sizing: border-box; margin: 0; padding: 10px 12px; border-radius: 10px; cursor: pointer;
  border: 1px solid #D5DEE7; background: #FFFFFF; transition: background .12s, border-color .12s;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {border-color: #E67E22; background: #FFF6EE;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
  background: #FDEBD9; border-color: #E67E22; box-shadow: inset 4px 0 0 #E67E22;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p {font-weight: 700; color: #16324F;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {display: none;}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p {font-size: 0.95rem; margin: 0;}
.st-key-month_pick div[role="radiogroup"] {gap: 8px; flex-wrap: wrap;}
.st-key-month_pick [data-testid="stRadioOption"] {padding: 8px 14px; border-radius: 999px; border: 1px solid #D5DEE7; background: #FFFFFF; cursor: pointer; margin: 0;}
.st-key-month_pick [data-testid="stRadioOption"]:hover {border-color: #E67E22; background: #FFF6EE;}
.st-key-month_pick [data-testid="stRadioOption"]:has(input:checked) {background: #E67E22; border-color: #E67E22;}
.st-key-month_pick [data-testid="stRadioOption"]:has(input:checked) p {color: #FFFFFF; font-weight: 700;}
.st-key-month_pick [data-testid="stRadioOption"] > div > div:first-child {display: none;}
.st-key-month_pick [data-testid="stRadioOption"] p {margin: 0;}
</style>"""


@st.cache_data(ttl=CONNECTION_RECHECK_SECONDS, show_spinner=False)
def model_info(api_url):
    return api_request(api_url, '/model-info', timeout=HEALTH_TIMEOUT)


@st.cache_data(ttl=60, show_spinner=False)
def inference_summary(api_url):
    return api_request(api_url, '/logs/summary', timeout=HEALTH_TIMEOUT)


@st.cache_resource(show_spinner=False)
def replay_scores():
    """Pre-computed late-risk and days-from-promise pairs for every replay order; never reads outcomes."""
    from deployment.app.orchestration import read_demo_json
    return read_demo_json('replay_scores.json')['scores']


def model_codes():
    """(late-risk, timing) model codes: from the live API once connected, else from the pre-scored replay file."""
    from deployment.app.orchestration import read_demo_json
    live = st.session_state.get('live_metadata')
    if live:
        return live['classification'].get('model_code', 'risk model'), live.get('regression', {}).get('model_code', 'timing model')
    codes = read_demo_json('replay_scores.json').get('model_codes', {})
    return codes.get('classification', 'risk model'), codes.get('regression', 'timing model')


@st.cache_resource(show_spinner=False)
def replay_score_percentiles():
    """Sorted replay late-risk scores for rank-based context; never reads outcomes."""
    from deployment.app.orchestration import read_demo_json
    scores = read_demo_json('replay_scores.json')['scores'].values()
    return sorted(pair[0] for pair in scores if pair is not None)


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
    if isinstance(exc, requests.Timeout):
        st.info('The prediction service may still be waking after inactivity. Select Retry API connection, then try again. Startup can take 50 seconds or more.')
    else:
        st.info('Check the API URL and select Retry API connection. After inactivity, the prediction service may take 50 seconds or more to wake.')


def page_header(title, lede):
    st.title(title)
    st.markdown(f'<p class="dw-lede">{lede}</p>', unsafe_allow_html=True)


def percent(value):
    return 'Not defined' if value is None else f'{value:.1%}'


def percentile_text(probability):
    replay = replay_score_percentiles()
    if not replay: return None
    return f'Higher than {bisect_left(replay, probability) / len(replay):.0%} of June–August 2018 orders'


def result_panel(result, stats, api_url, features):
    classification, regression = result['classification'], result['regression']
    st.subheader('Prediction results')
    probability = classification['late_probability']
    try:
        context = percentile_text(probability)
    except SimulationDataError:
        context = None
    risk, _ = model_codes()
    st.metric('Estimated late-delivery risk', f'{probability:.1%}', help=f'{risk} score used for ranking; not a calibrated probability.')
    st.progress(min(max(probability, 0.0), 1.0))
    threshold = stats.get('threshold')
    if threshold is not None:
        st.caption(f"{'Flagged' if probability >= threshold else 'Not flagged'} at the operating cut-off of {threshold:.1%} (the Validation F1-maximising cut-off).")
    if context: st.caption(f'{context}.')
    days = regression['predicted_days_from_promise']
    st.metric('Estimated days from promise', f'{days:+.2f} days')
    st.caption('Negative = arrives before the promised date.')
    st.caption('A low risk score does not guarantee on-time delivery.')
    for warning in classification.get('drift_warnings', []):
        st.warning(warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning))


def check_order(api_url, stats, samples):
    page_header('Single-order API demo', 'Score one held-out order with both models. Edit any input and predict again.')
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
    with right:
        if st.button('Predict delivery', type='primary', disabled=bool(errors), width='stretch'):
            try:
                with st.spinner('Scoring both models with the API…'):
                    result = api_request(api_url, '/predict', {'features': features})
                st.session_state['last_result'] = result
                st.session_state['last_features'] = features
            except (ValueError, requests.RequestException, SimulationDataError) as exc:
                show_error(exc)
        with st.container(border=True):
            if st.session_state.get('last_result'):
                result_panel(st.session_state['last_result'], stats, api_url, st.session_state['last_features'])
            else:
                st.subheader('Ready to predict')
                st.write('Results appear here after **Predict delivery**.')


def navigate_day(delta, minimum, maximum):
    current = st.session_state['simulation_day'] + timedelta(days=delta)
    # Clamp: queued clicks can arrive before the disabled state renders.
    st.session_state['simulation_day'] = min(max(current, minimum), maximum)


def risk_chart(frame, selected, risk):
    data = frame.assign(Group=[FOLLOW_LABEL if flag else OTHER_LABEL for flag in frame['follow_up_recommended']])
    bars = alt.Chart(data).mark_bar(width={'band': 0.9}).encode(
        x=alt.X('rank:Q', title='Risk rank (1 = highest risk)', scale=alt.Scale(domain=[0.5, len(data) + 0.5], nice=False)),
        y=alt.Y('late_probability:Q', title=f'{risk} late-delivery risk', axis=alt.Axis(format='%')),
        color=alt.Color('Group:N', scale=alt.Scale(domain=[FOLLOW_LABEL, OTHER_LABEL], range=[FOLLOW, OTHER]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('rank:Q', title='Rank'), alt.Tooltip('order_id:N', title='Order'),
                 alt.Tooltip('late_probability:Q', title=f'{risk} risk', format='.1%'), alt.Tooltip('Group:N', title='Decision')])
    layers = [bars]
    if 0 < selected < len(data):
        layers.append(alt.Chart(pd.DataFrame({'x': [selected + 0.5], 'label': [f'Follow-up cutoff: top {selected}']})).mark_rule(
            color=INK, strokeDash=[4, 3]).encode(x='x:Q', tooltip=alt.Tooltip('label:N', title='')))
    return alt.layer(*layers).properties(height=280)


def agreement_chart(frame, risk, timing):
    data = frame.assign(Group=[FOLLOW_LABEL if flag else OTHER_LABEL for flag in frame['follow_up_recommended']])
    points = alt.Chart(data).mark_circle(size=55, opacity=0.8).encode(
        x=alt.X('late_probability:Q', title=f'{risk} late-delivery risk', axis=alt.Axis(format='%')),
        y=alt.Y('predicted_days_from_promise:Q', title=f'{timing}: days before promised date'),
        color=alt.Color('Group:N', scale=alt.Scale(domain=[FOLLOW_LABEL, OTHER_LABEL], range=[FOLLOW, OTHER]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('order_id:N', title='Order'), alt.Tooltip('late_probability:Q', title=f'{risk} risk', format='.1%'),
                 alt.Tooltip('predicted_days_from_promise:Q', title=f'{timing} days', format='+.1f')])
    promise = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(y='y:Q')
    return alt.layer(promise, points).properties(height=280)


def score_progress():
    """Visible progress for the per-day API scoring instead of a silent wait."""
    slot = st.empty()
    def update(done, total):
        slot.progress(done / total if total else 1.0, text=f'Scoring orders with the prediction API… {done}/{total}')
    return slot, update


def daily_page(api_url, metadata):
    risk, timing = model_codes()
    page_header('Morning delivery follow-up', 'Each morning, yesterday’s approved orders are scored and the riskiest share is sent for follow-up.')
    config = configuration(load_demo_orders())
    minimum, maximum = date.fromisoformat(config['minimum_date']), date.fromisoformat(config['maximum_date'])
    current = st.session_state.get('simulation_day', date.fromisoformat(config['default_date']))
    st.session_state['simulation_day'] = min(max(current, minimum), maximum)
    day_box, policy_box = st.columns([1.15, 1])
    with day_box.container(border=True):
        previous, picker, following = st.columns([1, 1.5, 1], vertical_alignment='bottom')
        previous.button('Previous day', disabled=st.session_state['simulation_day'] <= minimum, on_click=navigate_day, args=(-1, minimum, maximum), width='stretch')
        day = picker.date_input('Simulated current date', min_value=minimum, max_value=maximum, key='simulation_day')
        following.button('Next day', disabled=day >= maximum, on_click=navigate_day, args=(1, minimum, maximum), width='stretch')
    with policy_box.container(border=True):
        review_col, remaining_col = st.columns([1.4, 1], vertical_alignment='bottom')
        review = review_col.slider('Follow-up proportion (%)', 0, 100, 30, step=5, help=f'Share of eligible orders, ranked by {risk} risk, sent for human follow-up. Rounds up; ties use order ID.')
        remaining = remaining_col.number_input('Min. days before promise', min_value=0, max_value=365, value=1, step=1, help='Orders whose promise is this many days away or fewer are excluded: too late to act on.')
    query = (api_url, model_versions(metadata), day.isoformat(), review, remaining, snapshot_revision())
    if st.session_state.get('worklist_policy') != query:
        st.session_state['worklist_policy'] = query
        st.session_state.pop('revealed_outcomes', None)
    slot, update = score_progress()
    result = build_day_worklist(api_url, model_versions(metadata), day, review / 100, remaining, snapshot_revision(), on_progress=update)
    slot.empty()
    st.subheader(f"Simulated today: {result['simulation_date']}")
    st.caption(f"Orders approved {result['batch_date']}, scored with information up to midnight.")
    cols = st.columns(4)
    labels = {'total': 'Orders in batch', 'eligible': 'Eligible orders', 'excluded': 'Excluded orders', 'selected': 'Selected for follow-up'}
    helps = {'total': 'Orders approved on the previous calendar day.', 'eligible': 'Still undelivered at the cutoff, with enough time before the promise.',
             'excluded': 'Already delivered, missing promise, or too close to the promise.', 'selected': f'Top {review}% of eligible orders by {risk} risk.'}
    for column, name in zip(cols, ['total', 'eligible', 'excluded', 'selected']):
        column.metric(labels[name], result['counts'][name], help=helps[name], border=True)
    reasons = result.get('exclusion_reasons', {})
    if reasons:
        with st.expander(f"Why {result['counts']['excluded']} orders were excluded"):
            st.caption('An order can have more than one reason.')
            st.dataframe(pd.DataFrame([{'Reason': k.replace('_', ' '), 'Orders': v} for k, v in reasons.items()]), hide_index=True, width='stretch')
    orders = result.get('orders', [])
    if orders:
        frame = pd.DataFrame(orders)
        chart_left, chart_right = st.columns(2)
        with chart_left.container(border=True):
            st.markdown('**Who gets a follow-up call?**')
            st.altair_chart(risk_chart(frame, result['counts']['selected'], risk), width='stretch')
            st.caption('One bar per order. Orange = today’s follow-up list.')
        with chart_right.container(border=True):
            st.markdown('**How the two models see each order**')
            st.altair_chart(agreement_chart(frame, risk, timing), width='stretch')
            st.caption(f'Each dot is an order. Right = higher {risk} risk. Higher = {timing} expects it to arrive closer to its promised date (dashed line), so less room for delay. Top-right orders worry both models.')
        columns = ['rank', 'order_id', 'late_probability', 'recommendation', 'promised_delivery_date', 'predicted_days_from_promise']
        display = frame[[c for c in columns if c in frame]].copy()
        display['promised_delivery_date'] = display['promised_delivery_date'].str[:10]
        display['recommendation'] = ['✅ Follow up' if flag else '—' for flag in frame['follow_up_recommended']]
        st.subheader('Risk-ranked eligible orders')
        top = float(display['late_probability'].max()) or 1.0
        st.dataframe(display, hide_index=True, width='stretch', height=320, column_config={
            'rank': st.column_config.NumberColumn('Rank', width='small'),
            'order_id': st.column_config.TextColumn('Order ID', width='medium'),
            'late_probability': st.column_config.ProgressColumn(f'{risk} risk', format='percent', min_value=0.0, max_value=top, width='medium'),
            'recommendation': st.column_config.TextColumn('Follow-up', width='small'),
            'promised_delivery_date': st.column_config.TextColumn('Promised date', width='small'),
            'predicted_days_from_promise': st.column_config.NumberColumn(f'{timing} days vs promise', format='%+.1f', width='small', help='Estimated days from promise: positive late, negative early.')})
        actions = st.columns([1.3, 2.4, 1.6], vertical_alignment='bottom')
        actions[0].download_button('Download prioritised worklist', display.to_csv(index=False), f'worklist_{day}.csv', 'text/csv', width='stretch')
        order_id = actions[1].selectbox('Inspect a ranked order', frame.order_id.tolist())
        selected = next(row for row in orders if row['order_id'] == order_id)
        if actions[2].button('Open selected order in API demo', width='stretch'):
            st.session_state['worklist_selected'] = selected
            st.session_state['open_order'] = order_id
            st.session_state['next_page'] = 'Single-order API demo'
            st.rerun()
        with st.expander('Selected order details', expanded=False):
            st.json(selected)
    else:
        st.info('No eligible orders for this day and policy. Try another day or review the eligibility assumptions.')
    with st.container(border=True):
        st.subheader('Retrospective evaluation')
        st.caption('Hidden by default: these outcomes were not known at the cutoff.')
        if st.button('Reveal eventual outcomes', key='reveal_outcomes'):
            st.session_state['revealed_outcomes'] = reveal_outcomes(result)
        if st.session_state.get('revealed_outcomes'):
            historical = st.session_state['revealed_outcomes']
            metrics = historical['metrics']
            st.warning('Retrospective outcomes: not known at prediction time.')
            summary = st.columns(3)
            summary[0].metric('Hit rate: followed-up orders that were late', percent(metrics.get('selected_late_rate')), border=True)
            summary[1].metric('Late orders caught (of eligible)', percent(metrics.get('eligible_late_coverage')), border=True)
            summary[2].metric('Late orders caught (of all)', percent(metrics.get('all_late_coverage')), border=True)
            st.dataframe(pd.DataFrame([{'Population': label, **metrics[label]} for label in ['all', 'eligible', 'selected']]), hide_index=True, width='stretch')
            st.caption('Unknown and canceled outcomes are excluded from rates. Cancellation is not known at approval, so canceled orders stay on the worklist.')
            values = historical.get('outcomes', {})
            with st.expander('Eventual outcome per order'):
                st.dataframe(pd.DataFrame([{'order_id': k, 'Eventual outcome': {0: 'On time', 1: 'Late', 'canceled': 'Canceled'}.get(v, 'Unknown')} for k, v in values.items()]), hide_index=True, width='stretch')
    with st.expander('Simulation assumptions and limits'):
        st.write(f"Model available from {config.get('model_available_date', minimum)}. Dates are restricted to the supported replay period.")
        st.write('Eligible promises must be strictly more than the selected minimum days away. The follow-up proportion and minimum days are configurable operational assumptions, not a validated universal policy.')
        for limitation in config.get('limitations', []): st.write(limitation)


MONTH_NOTES = {
    '2018-06': ('Test month', 'The models were fitted on history before 2 June and chosen on Validation: June is their held-out test month.'),
    '2018-07': ('Monitoring month', 'The models were frozen on 2 June and never saw July: this is the true hold-out.'),
    '2018-08': ('Monitoring month', 'The models were frozen on 2 June and never saw August: this is the true hold-out.'),
}


@st.cache_data(show_spinner=False, max_entries=24)
def month_results(revision, minimum_days_remaining, month):
    scored, outcomes, timing = load_month_inputs()
    data = load_demo_orders()
    replay = month_replay(data, scored['scores'], outcomes, minimum_days_remaining, month)
    # Comparison only: the deployed list ranks by late risk; this shows the same daily rules ranked by the timing model's buffer.
    buffer_replay = month_replay(data, buffer_ranking_scores(scored['scores']), outcomes, minimum_days_remaining, month)
    curve = []
    for share in range(0, 101, 5):
        daily = month_summary(replay, share / 100)['pooled']
        monthly = monthly_ranking(data, scored['scores'], outcomes, share / 100, month)
        by_buffer = month_summary(buffer_replay, share / 100)['pooled']
        curve.append({'share': share, 'daily': daily['all_late_coverage'], 'monthly': monthly['all_late_coverage'], 'buffer': by_buffer['all_late_coverage'],
                      'daily_caught': daily['late_caught'], 'buffer_caught': by_buffer['late_caught'], 'late_all': daily['late_all']})
    return {'replay': replay, 'curve': curve, 'models': scored.get('models', {}), 'generated_at': scored.get('generated_at_utc'),
            'timing': timing_accuracy(data, scored['scores'], timing, outcomes, month)}


def cumulative_chart(days):
    frame = pd.DataFrame(days)
    frame['date'] = pd.to_datetime(frame['simulation_date'])
    frame['All late orders'] = frame['late_all'].cumsum()
    frame['Caught by follow-up'] = frame['late_caught'].cumsum()
    long = frame.melt(id_vars='date', value_vars=['All late orders', 'Caught by follow-up'], var_name='Series', value_name='Late orders')
    return alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('date:T', title=None, axis=alt.Axis(format='%d %b')),
        y=alt.Y('Late orders:Q', title='Late orders (running total)'),
        color=alt.Color('Series:N', scale=alt.Scale(domain=['All late orders', 'Caught by follow-up'], range=[OTHER, FOLLOW]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('date:T', title='Day', format='%d %b %Y'), 'Series:N', 'Late orders:Q']).properties(height=300)


def daily_bars_chart(days):
    frame = pd.DataFrame(days)
    frame['date'] = pd.to_datetime(frame['simulation_date'])
    frame['Caught'] = frame['late_caught']
    frame['Missed'] = frame['late_all'] - frame['late_caught']
    long = frame.melt(id_vars=['date', 'selected'], value_vars=['Caught', 'Missed'], var_name='Late orders', value_name='Orders')
    long['stack'] = long['Late orders'].map({'Caught': 0, 'Missed': 1})
    return alt.Chart(long).mark_bar().encode(
        x=alt.X('date:T', title=None, axis=alt.Axis(format='%d %b')),
        y=alt.Y('Orders:Q', title='Late orders that day', stack=True),
        order=alt.Order('stack:Q'),
        color=alt.Color('Late orders:N', scale=alt.Scale(domain=['Caught', 'Missed'], range=[FOLLOW, OTHER]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('date:T', title='Day', format='%d %b %Y'), 'Late orders:N', 'Orders:Q', alt.Tooltip('selected:Q', title='Orders followed up')]).properties(height=300)


def tradeoff_chart(curve, share, current, risk, timing):
    rows = []
    for point in curve:
        rows += [{'share': point['share'], 'Policy': f'{risk} daily list (deployed)', 'caught': point['daily']},
                 {'share': point['share'], 'Policy': f'{timing} buffer daily list', 'caught': point['buffer']},
                 {'share': point['share'], 'Policy': f'{risk} whole-month ranking', 'caught': point['monthly']},
                 {'share': point['share'], 'Policy': 'Random', 'caught': point['share'] / 100}]
    domain = [f'{risk} daily list (deployed)', f'{timing} buffer daily list', f'{risk} whole-month ranking', 'Random']
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2.5).encode(
        x=alt.X('share:Q', title='Share of each day’s orders you follow up (%)', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('caught:Q', title='Share of all late deliveries caught', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Policy:N', scale=alt.Scale(domain=domain, range=[FOLLOW, TEAL, INK, OTHER]), legend=alt.Legend(orient='top', title=None)),
        strokeDash=alt.StrokeDash('Policy:N', scale=alt.Scale(domain=domain, range=[[1, 0], [1, 0], [6, 3], [2, 2]]), legend=None),
        tooltip=[alt.Tooltip('share:Q', title='Followed up %'), 'Policy:N', alt.Tooltip('caught:Q', title='Late caught', format='.1%')])
    point = pd.DataFrame({'share': [share], 'caught': [current or 0],
                          'label': [f'Call {share}% of orders → catch {percent(current)} of late deliveries']})
    marker = alt.Chart(point).mark_point(size=160, filled=True, color=FOLLOW, stroke=INK, strokeWidth=1.5).encode(
        x='share:Q', y='caught:Q', tooltip=alt.Tooltip('label:N', title=''))
    return alt.layer(lines, marker).properties(height=320)


def timing_chart(days, timing):
    frame = pd.DataFrame(days)
    frame['date'] = pd.to_datetime(frame['approval_date'])
    frame = frame.rename(columns={'actual': 'Actual arrival', 'predicted': f'{timing} prediction'})
    long = frame.melt(id_vars=['date', 'n'], value_vars=['Actual arrival', f'{timing} prediction'], var_name='Series', value_name='Days')
    lines = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('date:T', title='Approval day', axis=alt.Axis(format='%d %b')),
        y=alt.Y('Days:Q', title='Days from promise (− early, + late)'),
        color=alt.Color('Series:N', scale=alt.Scale(domain=['Actual arrival', f'{timing} prediction'], range=[INK, FOLLOW]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('date:T', title='Approved', format='%d %b %Y'), 'Series:N', alt.Tooltip('Days:Q', format='+.1f'), alt.Tooltip('n:Q', title='Orders')])
    promise = alt.Chart(pd.DataFrame({'y': [0], 'label': ['Promised date']})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(
        y='y:Q', tooltip=alt.Tooltip('label:N', title=''))
    return alt.layer(promise, lines).properties(height=300)


def month_page(live_metadata):
    risk, timing_code = model_codes()
    header = st.container()
    month = st.radio('Month', list(REPLAY_MONTHS), index=list(REPLAY_MONTHS).index('2018-07'), format_func=lambda key: f'{REPLAY_MONTHS[key]} · {MONTH_NOTES[key][0]}',
                     horizontal=True, key='month_pick', label_visibility='collapsed')
    name, (badge, note) = REPLAY_MONTHS[month], MONTH_NOTES[month]
    with header:
        page_header(f'Month-end review · {name}', f'Late deliveries the daily list would have caught, and how accurate {timing_code} was.')
    st.info(f'**{badge}.** {note} Retrospective view: uses outcomes not known on the day.')
    controls = st.columns([2, 1.4, 2.6], vertical_alignment='bottom')
    share = controls[0].slider('Follow-up proportion (%)', 0, 100, 30, step=5, key='month_share', help='Applied every day to that day’s eligible orders, rounding up.')
    remaining = controls[1].number_input('Min. days before promise', min_value=0, max_value=30, value=1, step=1, key='month_remaining', help='Orders whose promise is this many days away or fewer are excluded: too late to act on.')
    results = month_results(snapshot_revision(), remaining, month)
    summary = month_summary(results['replay'], share / 100)
    pooled = summary['pooled']
    monthly = next(point['monthly'] for point in results['curve'] if point['share'] == share)
    base_rate = pooled['late_all'] / pooled['known_all'] if pooled['known_all'] else None
    hit_rate = pooled['selected_late_rate']
    st.subheader(f'{risk} follow-up: late deliveries caught')
    cards = st.columns(4)
    cards[0].metric('Orders followed up', f"{pooled['selected']:,}", f"{pooled['selected'] / pooled['total']:.0%} of {pooled['total']:,} orders" if pooled['total'] else None, delta_color='off', delta_arrow='off', border=True)
    cards[1].metric('Late deliveries caught', f"{pooled['late_caught']} of {pooled['late_all']}", f"{percent(pooled['all_late_coverage'])} captured", delta_color='off', delta_arrow='off', border=True,
                    help='Known late orders among those followed up, out of every known late order that month (including excluded ones). Canceled orders have no outcome and never count as late.')
    cards[2].metric('Hit rate of follow-ups', percent(hit_rate), f"{hit_rate / base_rate:.1f}× the {base_rate:.1%} late rate" if hit_rate and base_rate else None, delta_color='off', delta_arrow='off', border=True,
                    help='Share of followed-up orders (with known outcomes) that were eventually late. Random follow-up would match the month’s late rate.')
    delta = None if pooled['all_late_coverage'] is None or monthly is None else pooled['all_late_coverage'] - monthly
    cards[3].metric('Whole-month ranking', percent(monthly), None if delta is None else f'{delta * 100:+.1f} pts for the daily list', border=True,
                    help='Phase 2 style: rank the whole month’s orders at once and take the same top share. Needs the whole month in hand, so it is not available on the day.')
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown(f'**Late deliveries caught through {name.split()[0]}**')
        st.altair_chart(cumulative_chart(summary['days']), width='stretch')
        st.caption('The gap is late orders the list missed.')
    with right.container(border=True):
        st.markdown('**Caught vs missed, day by day**')
        st.altair_chart(daily_bars_chart(summary['days']), width='stretch')
    with st.container(border=True):
        st.markdown('**How much follow-up effort buys how many catches?**')
        st.caption(f"Your setting (dot): call {share}% of orders → catch {percent(pooled['all_late_coverage'])} of late deliveries.")
        st.altair_chart(tradeoff_chart(results['curve'], share, pooled['all_late_coverage'], risk, timing_code), width='stretch')
        point = next(p for p in results['curve'] if p['share'] == share)
        if point['late_all']:
            st.info(f"At {share}%, {risk} caught {point['daily_caught']} of {point['late_all']} late deliveries; "
                    f"ranking by {timing_code}’s buffer would have caught {point['buffer_caught']}.")
    timing = results['timing']
    if timing:
        st.subheader(f'{timing_code} timing: how close were the arrival estimates?')
        june_rmse = ((live_metadata or {}).get('regression') or {}).get('evaluation', {}).get('June', {}).get('rmse')
        cards = st.columns(4)
        cards[0].metric('Typical error (MAE)', f"{timing['mae']:.1f} days", help=f'Mean absolute gap between {timing_code}’s predicted and the actual days from promise.', border=True)
        cards[1].metric('RMSE', f"{timing['rmse']:.2f} days", help='Penalises large misses more.' + (f' Phase 2 reported {june_rmse:.3f} days for June.' if june_rmse else ''), border=True)
        cards[2].metric(f'Actual vs {timing_code} (average days)', f"{timing['mean_actual']:+.1f} vs {timing['mean_predicted']:+.1f}",
                        f"{timing_code} {'later' if timing['bias'] > 0 else 'earlier'} by {abs(timing['bias']):.1f} days on average", delta_color='off', delta_arrow='off', border=True,
                        help=f'Days from promise (negative = early). {timing_code} erring later is the cautious direction for operations.')
        if timing['predicted_late_orders'] is not None and timing['predicted_on_time_orders'] is not None:
            cards[3].metric(f'{timing_code} buffer: late vs on-time', f"{timing['predicted_late_orders']:+.1f} vs {timing['predicted_on_time_orders']:+.1f}",
                            'Late orders: thinner buffer', delta_color='off', delta_arrow='off', border=True,
                            help='Average timing-model prediction for orders that were eventually late vs on time. A smaller buffer for late orders shows the timing model carries a risk signal.')
        with st.container(border=True):
            st.markdown('**Predicted vs actual days from promise, by approval day**')
            st.altair_chart(timing_chart(timing['days'], timing_code), width='stretch')
            st.caption('Negative = before the promised date. A prediction line above the actual line errs late, the cautious side.')
    table = pd.DataFrame(summary['days']).rename(columns={'simulation_date': 'Simulated day', 'total': 'Orders', 'eligible': 'Eligible', 'selected': 'Followed up',
                                                          'late_caught': 'Late caught', 'late_eligible': 'Late (eligible)', 'late_all': 'Late (all)'}).drop(columns=['selected_known', 'known_all'])
    with st.expander('Day-by-day table'):
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Download month-end table', table.to_csv(index=False), f'month_end_{month}_{share}pct.csv', 'text/csv')
    with st.expander('How these numbers are produced'):
        st.write('Each morning uses the daily list rules; outcomes are attached only after ranking.')
        st.write(f"Scores are pre-computed with the same {risk} and {timing_code} models the API serves (`python -m deployment.scripts.score_replay`), so this page works without the API.")
        st.write('Late = Phase 2 label within 45 days of approval. Canceled and unavailable orders stay in the daily list (cancellation is unknown at approval) but have no outcome, so they never count as late or toward rates. Timing accuracy uses the Phase 2 target and excludes them.')
        st.caption(f"Pre-scored models: {', '.join(f'{k} {v}' for k, v in results['models'].items())}")
    if live_metadata is not None:
        live = dict(model_versions(live_metadata))
        if any(live.get(name) != version for name, version in results['models'].items()):
            st.warning(f'The live API now serves different model versions ({live}). Rerun python -m deployment.scripts.score_replay so this page matches.')


def scored_upload_frame(incoming, response):
    """Keep original inputs beside API results so a downloaded CSV can be rescored."""
    output = []
    for item in response['results']:
        prediction = item.get('prediction') or {}
        output.append({**prediction.get('classification', {}), **prediction.get('regression', {}), 'error': item.get('error')})
    return pd.concat([incoming, pd.DataFrame(output)], axis=1)


def batch_upload(api_url, stats):
    page_header('Batch API demo', 'Score a CSV of orders with both models.')
    st.caption('Use the template columns; order_id is optional. Blank numbers count as missing. Up to 1,000 rows.')
    feature_cols = stats['features']['numeric'] + stats['features']['categorical']
    st.download_button('Download feature template', pd.DataFrame(columns=['order_id'] + feature_cols).to_csv(index=False), 'feature_template.csv', 'text/csv')
    upload = st.file_uploader('Engineered order features', type=['csv'])
    if upload is not None and st.button('Score uploaded orders', type='primary'):
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


def selection_table(records, score_name, digits):
    """Validation model-choice table from the exported selection records."""
    rows = [{'Code': r['Code'], 'Model': r['Model'], score_name: r['validation'], 'Difference vs leader': r['diff_vs_leader'],
             '95% interval': f"{r['ci_low']:+.{digits}f} to {r['ci_high']:+.{digits}f}", 'Tied with leader': 'Yes' if r['tied'] else 'No',
             'Decision': r['decision']} for r in records]
    return pd.DataFrame(rows)


def choice_section(stats, regression, risk, timing):
    """Why these models: the Validation choice, then the classifier's cut-off and its June misses."""
    st.subheader('How the models were chosen')
    st.write('Candidates were tuned by cross-validation inside each family. Only the Validation set chooses between them, and June never does. '
             f"{stats.get('selection_rule', '')}")
    left, right = st.columns(2)
    for column, (title, meta, score, digits) in zip((left, right), [('Late-delivery risk: Validation average precision (higher is better)', stats, 'Average precision', 4),
                                                                    ('Days from promise: Validation RMSE in days (lower is better)', regression, 'RMSE (days)', 4)]):
        with column.container(border=True):
            st.markdown(f'**{title}**')
            if meta.get('selection'):
                st.dataframe(selection_table(meta['selection'], score, digits), hide_index=True, width='stretch')
                if meta.get('selected_by_order_bootstrap') not in (None, meta.get('model_code')):
                    st.caption(f"Resampling single orders instead of days would pick {meta['selected_by_order_bootstrap']}.")
            else:
                st.info('Selection table unavailable for this model version.')
    threshold = stats.get('threshold')
    if not stats.get('confusion') or threshold is None:
        return
    st.subheader(f'{risk} flags: cut-off and misses')
    st.write(f"The operating cut-off is {threshold:.1%}: the probability that maximises F1 on Validation for the Train-only fit, applied unchanged to June. "
             'Scores are for ranking; the default 0.5 flags almost nothing because late orders are rare.')
    confusion = pd.DataFrame(stats['confusion']).rename(columns={
        'split': 'Split', 'cut-off': 'Cut-off', 'true_positive': 'Late, flagged', 'false_negative': 'Late, missed',
        'false_positive': 'On time, flagged', 'true_negative': 'On time, not flagged', 'flagged_pct': 'Flagged (%)',
        'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})
    st.dataframe(confusion[['Split', 'Cut-off', 'Late, flagged', 'Late, missed', 'On time, flagged', 'On time, not flagged', 'Flagged (%)', 'Precision', 'Recall', 'F1']],
                 hide_index=True, width='stretch', column_config={
                     'Flagged (%)': st.column_config.NumberColumn(format='%.1f'), 'Precision': st.column_config.NumberColumn(format='percent'),
                     'Recall': st.column_config.NumberColumn(format='percent'), 'F1': st.column_config.NumberColumn(format='%.3f')})
    st.caption('Known outcomes only; canceled orders are excluded.')
    misses = pd.DataFrame(stats.get('false_negatives', []))
    if not misses.empty:
        misses = misses[misses['cut_off'] != '0.5']
        with st.container(border=True):
            st.markdown(f'**June late orders {risk} missed at the operating cut-off**')
            tabs = st.tabs(['Customer state', 'Route type', 'Promised lead time'])
            for tab, breakdown in zip(tabs, ['customer_state', 'route_type', 'promised_lead_band']):
                shown = misses[misses['breakdown'] == breakdown].head(12).rename(columns={
                    'group': 'Group', 'late_orders': 'Late orders', 'missed': 'Missed', 'miss_rate_pct': 'Miss rate (%)', 'share_of_missed_pct': 'Share of all misses (%)'})
                tab.dataframe(shown[['Group', 'Late orders', 'Missed', 'Miss rate (%)', 'Share of all misses (%)']], hide_index=True, width='stretch',
                              column_config={'Miss rate (%)': st.column_config.NumberColumn(format='%.1f'), 'Share of all misses (%)': st.column_config.NumberColumn(format='%.1f')})


def guidance(metadata, api_url=None):
    risk, timing = model_codes()
    page_header('Models and guidance', f'How {risk} and {timing} were chosen and how they held up.')
    stats, regression = metadata['classification'], metadata.get('regression', {})
    numeric, categorical = stats['features']['numeric'], stats['features']['categorical']
    def used(meta):
        return len(meta.get('selected_features') or numeric + categorical)
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown(f'**{risk} · late-delivery risk** ({short_label(risk)})')
        st.write('Ranks orders by late-delivery risk. Scores are for ranking, not calibrated probabilities.')
        st.caption(f"Version {stats.get('model_version', '?')} · {stats.get('train_rows', 0):,} training orders known before 2 June 2018 · {used(stats)} of {len(numeric) + len(categorical)} features.")
    with right.container(border=True):
        st.markdown(f'**{timing} · days from promise** ({short_label(timing)})')
        st.write('Predicts days before (−) or after (+) the promised date.')
        st.caption(f"Version {regression.get('model_version', '?')} · {used(regression)} of {len(numeric) + len(categorical)} features · frozen with {risk} on 2 June 2018.")
    facts = st.columns(4)
    facts[0].metric(f'{risk} training orders', f"{stats.get('train_rows', 0):,}", border=True)
    facts[1].metric(f'{timing} training orders', f"{regression.get('train_rows', 0):,}", border=True)
    facts[2].metric('Input features', len(numeric) + len(categorical), f'{len(numeric)} numeric · {len(categorical)} categorical', delta_color='off', delta_arrow='off', border=True)
    facts[3].metric('Replay orders (Jun–Aug 2018)', f'{len(replay_scores()):,}', 'held out: never used in training', delta_color='off', delta_arrow='off', border=True)
    with st.expander('Features used (all known at payment approval)'):
        groups = st.columns(len(GROUPS))
        for column, (title, names) in zip(groups, GROUPS.items()):
            column.markdown(f'**{title}**')
            column.markdown('\n'.join(f'- {LABELS[name][0]}' for name in names))
    choice_section(stats, regression, risk, timing)
    st.subheader('How the models held up, month by month')
    rows = []
    for month, name in REPLAY_MONTHS.items():
        c3 = stats.get('evaluation', {}).get(name.split()[0], {})
        try:
            results = month_results(snapshot_revision(), 1, month)
        except SimulationDataError:
            results = None
        timing_scores = results['timing'] if results else None
        point = next((p for p in results['curve'] if p['share'] == 30), None) if results else None
        rows.append({'Month': f"{name.split()[0]} · {'test' if month == '2018-06' else 'monitoring'}", 'Late rate': c3.get('late_rate'),
                     f'{risk} AUC': c3.get('roc_auc'), f'{risk} AP': c3.get('avg_precision'),
                     f'Caught @30% · {risk}': point['daily'] if point else None, f'Caught @30% · {timing} buffer': point['buffer'] if point else None,
                     f'{timing} MAE': timing_scores['mae'] if timing_scores else None, f'{timing} RMSE': timing_scores['rmse'] if timing_scores else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_config={
        'Late rate': st.column_config.NumberColumn(format='percent'), f'{risk} AUC': st.column_config.NumberColumn(format='%.3f', help='ROC-AUC; 0.5 = random ranking.'),
        f'{risk} AP': st.column_config.NumberColumn(format='percent', help='Average precision.'),
        f'Caught @30% · {risk}': st.column_config.NumberColumn(format='percent'), f'Caught @30% · {timing} buffer': st.column_config.NumberColumn(format='percent'),
        f'{timing} MAE': st.column_config.NumberColumn(format='%.2f d'), f'{timing} RMSE': st.column_config.NumberColumn(format='%.2f d')})
    st.caption('AUC 0.5 = random ranking. “Caught @30%” uses the daily list rules.')
    with st.container(border=True):
        st.markdown('**What monitoring tells us**')
        first, last = rows[0], rows[-1]
        notes = []
        if first['Late rate'] is not None and last['Late rate'] is not None:
            notes.append(f"Late deliveries went from {first['Late rate']:.1%} (June) to {last['Late rate']:.1%} (August) of orders with a known outcome.")
        if first[f'{risk} AUC'] is not None and last[f'{risk} AUC'] is not None:
            notes.append(f"{risk}’s AUC went from {first[f'{risk} AUC']:.2f} to {last[f'{risk} AUC']:.2f}.")
        if first[f'{timing} RMSE'] is not None and last[f'{timing} RMSE'] is not None:
            notes.append(f"{timing}’s RMSE went from {first[f'{timing} RMSE']:.1f} to {last[f'{timing} RMSE']:.1f} days.")
        if len(notes) == 3 and first[f'{risk} AUC'] > last[f'{risk} AUC']:
            notes.append(f'Next step: retrain {risk}, or rank with {timing}’s buffer.')
        st.write('\n'.join(f'- {note}' for note in notes) or 'No monitoring results are available.')
    with st.container(border=True):
        st.markdown('**How to use the scores**')
        st.write(f'- Call the top share of orders by {risk} risk that your team can handle.\n'
                 f'- {timing} close to 0 days means little slack before the promise.\n'
                 '- A low score does not guarantee on-time delivery.')
    st.subheader('Live inference monitoring')
    try:
        live = inference_summary(api_url) if api_url else None
    except (ValueError, requests.RequestException):
        live = None
    if live:
        cols = st.columns(4)
        cols[0].metric('API prediction requests', f"{live.get('request_count', 0):,}", border=True, help='Direct prediction calls since the Render service last started.')
        cols[1].metric('Orders scored', f"{live.get('prediction_count', 0):,}", border=True, help=f'Successful {risk} classifications, including each row of a batch.')
        mean = live.get('mean_score')
        cols[2].metric(f'Mean {risk} risk (recent window)', '—' if mean is None else f'{mean:.1%}', f"last {live.get('window_size', 0)} scores", delta_color='off', delta_arrow='off', border=True)
        warnings_seen = len(live.get('recent_warnings', []))
        cols[3].metric('Recent drift warnings', f"{warnings_seen}+" if warnings_seen >= 20 else warnings_seen, help='Recent predictions with an input outside the training 1st–99th percentile.', border=True)
        st.caption('From the API’s /logs/summary. Resets when the Render service restarts.')
    else:
        st.info('Live monitoring needs the prediction API. Select Retry API connection if it is waking.')
    st.caption('Built on 2018 data; may not reflect current operations.')
    with st.expander('Model metadata and input schema'):
        st.json(metadata)


# Sidebar order: what each page shows, then what to do there.
GUIDE = [
    ('Daily simulation',
     'One morning’s follow-up list. Orders approved the day before are scored by the live API and ranked by late-delivery risk; the riskiest share is marked for a call.',
     'It opens on 2 June 2018, the first scoring morning for the frozen models, covering 1 June approvals in the test month; any day from 2 June to 30 August works. Optionally change the share to call. Press **Reveal eventual outcomes** to see which orders were actually late.'),
    ('Month-end review',
     'The same daily list replayed for a whole month: how many late deliveries it caught, how effort trades off against catches, and how accurate the timing model was.',
     'It opens on July. Compare July and August (monitoring months the models never saw) with June (their held-out test month).'),
    ('Explore the data',
     'The data behind the models, from all 98,958 approved orders (January 2017 to August 2018): what drives late deliveries, plus our other two candidate problems (repeat purchases and reviews) and the models’ outputs.',
     'Filter by month, state, route or category, then move through the tabs.'),
    ('Single-order API demo',
     'One order sent to the live API, showing both predictions: late-delivery risk and days from the promised date.',
     'Pick an order, change any of its 24 inputs and press **Predict delivery**.'),
    ('Batch API demo',
     'Scores many orders in one API call, for your own data.',
     'Download the template, fill in up to 1,000 orders and upload it.'),
    ('Models and guidance',
     'How the two models were built, how they performed month by month, how to read their scores, and live API usage.',
     'Read before relying on the scores.'),
]


def start_page():
    risk, timing = model_codes()
    page_header('Delivery Watch', 'Ranks yesterday’s Olist orders by late-delivery risk, so operations know which to call first.')
    st.subheader('How to use this app')
    st.caption('Work through the pages in the order they appear in the sidebar.')
    for number, (name, shows, action) in enumerate(GUIDE, 1):
        with st.container(border=True):
            st.markdown(f'**{number}. {PAGE_ICONS[name]}\u2002{name}**  \n{shows}  \n**What to do:** {action}')
    with st.expander('Good to know'):
        st.markdown(
            f'- **{risk}** ({short_label(risk)}) ranks orders by late-delivery risk; its scores are for ranking, not calibrated probabilities.\n'
            f'- **{timing}** ({short_label(timing)}) predicts days from the promised date: negative = early, positive = late.\n'
            '- Data: real Olist orders from 2017–2018, replayed as if new. The models were fitted on history before 2 June and chosen on Validation; June is their test month and July and August are unseen.\n'
            '- The prediction API sleeps when idle; the first call can take 30–60 seconds. Use **Retry API connection** in the sidebar if needed.\n'
            '- API documentation: https://team8-delivery-api.onrender.com/docs')


def connect(api_url, force=False):
    """Check the API once per session (and on Retry), polling briefly with visible progress instead of one long call."""
    checked = st.session_state.get('connection_checked')
    if not force and checked and checked[0] == api_url and time.monotonic() - checked[1] < CONNECTION_RECHECK_SECONDS:
        return model_info(api_url)
    slot = st.empty()
    with slot.status('Checking / waking the prediction service. After inactivity this may take 50 seconds or more…', expanded=False) as status:
        def waiting(elapsed, error):
            status.update(label=f'Waiting for the prediction service to wake… {elapsed:.0f}s of up to {WAKE_BUDGET_SECONDS}s. Month-end review works meanwhile.')
        wait_for_service(api_url, on_wait=waiting)
        metadata = model_info(api_url)
    slot.empty()
    st.session_state['connection_checked'] = (api_url, time.monotonic())
    return metadata


def main():
    st.set_page_config(page_title='Delivery Watch', page_icon='📦', layout='wide')
    st.markdown(STYLE, unsafe_allow_html=True)
    api_url = os.environ.get('DELIVERY_API_URL', os.environ.get('CLASSIFICATION_API_URL', 'http://127.0.0.1:8002'))
    st.sidebar.title('📦\u2002Delivery Watch')
    st.sidebar.caption('Olist orders · 2017–2018')
    if st.session_state.get('next_page'):
        st.session_state['workspace'] = st.session_state.pop('next_page')
    page = st.sidebar.radio('Workspace', PAGES, key='workspace', label_visibility='collapsed',
                            format_func=lambda name: f'{PAGE_ICONS[name]}\u2002 {name}')
    st.sidebar.divider()
    status_area = st.sidebar.container()
    st.sidebar.caption('After inactivity, the API can take 50 seconds or more to wake.')
    retry = st.sidebar.button('Retry API connection', width='stretch')
    if retry:
        model_info.clear()
    metadata = None
    if page in API_PAGES or retry:
        try:
            metadata = connect(api_url, force=retry)
            service_identity = (api_url, model_versions(metadata))
            if st.session_state.get('prediction_service') != service_identity:
                st.session_state.pop('last_result', None)
                st.session_state.pop('last_features', None)
                st.session_state['prediction_service'] = service_identity
            st.session_state['api_connected'] = True
            st.session_state['live_metadata'] = metadata
            status_area.success(f"API connected · {' + '.join(model_codes())}")
        except (ValueError, requests.RequestException, SimulationDataError) as exc:
            st.session_state['api_connected'] = False
            status_area.error('API unavailable')
            st.session_state.pop('connection_checked', None)
            show_error(exc)
            st.sidebar.caption(api_url)
            if page in API_PAGES:
                st.info('Month-end review works without the API. Select it in the sidebar while the service wakes.')
                return
    else:
        # Offline-capable pages reuse the last metadata and never wait on the network.
        metadata = st.session_state.get('live_metadata')
        if st.session_state.get('api_connected'): status_area.success(f"API connected · {' + '.join(model_codes())}")
        else: status_area.caption('This page works without the prediction API.')
    st.sidebar.caption(api_url)
    with st.sidebar.expander('About this app'):
        st.markdown('**Frontend:** Streamlit — daily replay, ranking, charts.\n\n'
                    '**Backend:** FastAPI on Render — late-delivery risk and days-from-promise predictions.\n\n'
                    '**Data:** held-out Olist orders, June–August 2018.')
    try:
        if page == 'Start here': start_page()
        elif page == 'Daily simulation': daily_page(api_url, metadata)
        elif page == 'Month-end review': month_page(metadata)
        elif page == 'Explore the data': explore_page(page_header, replay_scores(), model_codes())
        elif page == 'Single-order API demo':
            data = load_demo_samples()
            samples = data.get('samples', []) if isinstance(data, dict) else data
            extra = st.session_state.get('worklist_selected')
            if extra and extra['order_id'] not in [r['order_id'] for r in samples]: samples.append(extra)
            check_order(api_url, metadata['classification'], samples)
        elif page == 'Batch API demo': batch_upload(api_url, metadata['classification'])
        else: guidance(metadata, api_url)
    except (ValueError, requests.RequestException, SimulationDataError) as exc:
        show_error(exc)


if __name__ == '__main__': main()
