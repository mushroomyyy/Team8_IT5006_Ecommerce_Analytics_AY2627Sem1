"""Operations proof of concept; all model operations use the classification HTTP API."""
from __future__ import annotations

import calendar
import io
import json
import math
import os

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


def api_request(api_url, endpoint, payload=None):
    response = (requests.get(f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}", timeout=60)
                if payload is None else requests.post(f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}", json=payload, timeout=60))
    if not response.ok:
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f'API returned {response.status_code}: {detail}')
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
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


def result_panel(result, stats, api_url, features):
    probability = result['late_probability']
    distribution = stats.get('june_score_distribution', {})
    boundary = distribution.get('top_10_percent_boundary')
    st.subheader('Order watch-list signal')
    st.caption('Result for the last submitted inputs. Submit again after editing the form.')
    st.metric('Estimated late-delivery probability', f'{probability:.1%}')
    st.progress(min(1.0,max(0.0,probability)))
    scores = distribution.get('sorted_scores',[])
    if scores:
        percentile = sum(s <= probability for s in scores) / len(scores)
        st.write(f'Higher than approximately **{percentile:.0%} of June orders**.')
    if boundary is not None:
        st.info(f'{"Would enter" if probability >= boundary else "Below"} the historical June top-10% watch-list boundary ({boundary:.1%}). This is a June reference, not today’s cohort ranking.')
    base_rate = distribution.get('late_rate', stats.get('evaluation',{}).get('June',{}).get('base_rate'))
    if base_rate:
        st.caption(f'Historical June late rate: {base_rate:.1%}; this score is {probability/base_rate:.1f}× that reference rate.')
    st.write('Prioritise a fulfilment check and review the delivery promise.' if boundary is not None and probability >= boundary else 'Continue routine monitoring; a low score does not guarantee on-time delivery.')
    st.caption('Simulation using held-out 2018 orders. The 0.5 cutoff is illustrative and is not an operational policy.')
    for warning in result.get('drift_warnings',[]):
        st.warning(warning.get('message',str(warning)) if isinstance(warning,dict) else str(warning))
    histogram = distribution.get('histogram',[])
    if histogram:
        bars = alt.Chart(pd.DataFrame(histogram)).mark_bar(color='#B9CDD9').encode(x=alt.X('lower:Q', title='Predicted late-delivery probability', axis=alt.Axis(format='%')), x2='upper:Q',y=alt.Y('count:Q',title='Historical June orders'))
        mark = alt.Chart(pd.DataFrame({'score':[probability]})).mark_rule(color='#E67E22',strokeWidth=3).encode(x='score:Q')
        st.altair_chart(bars+mark, width='stretch')
    if st.button('Why this score?', key='explain_score'):
        try:
            with st.spinner('Comparing each feature with its training baseline…'):
                explanation = api_request(api_url,'/explain',{'features':features})
                st.session_state['explanation'] = explanation
        except (ValueError, requests.RequestException) as exc: st.error(str(exc))
    if st.session_state.get('explanation'):
            explanation=st.session_state['explanation']
            drivers = explanation.get('drivers', explanation.get('sensitivities',[]))
            st.caption('Sensitivity check: each feature is reset to its training median (categoricals to mode). This is not causal and correlated features can affect interpretation.')
            st.dataframe(pd.DataFrame(drivers),width='stretch')
            if drivers:
                frame = pd.DataFrame(drivers)
                value_col = next((c for c in ['probability_change','delta','change'] if c in frame),None)
                if value_col:
                    st.altair_chart(alt.Chart(frame).mark_bar(color='#E67E22').encode(x=alt.X(f'{value_col}:Q',title='Probability change versus baseline reset'),y=alt.Y('feature:N',sort='-x')),width='stretch')


def check_order(api_url,stats,samples):
    st.title('Check an order')
    st.write('For the operations and logistics team: prioritise newly approved orders for follow-up. Choose a held-out June 2018 order, review its inputs, then check its risk.')
    if st.session_state.get('open_order'):
        st.session_state['sample_picker'] = st.session_state.pop('open_order')
    sample_id = st.selectbox('Held-out historical order',[r['order_id'] for r in samples], key='sample_picker')
    sample = next(r for r in samples if r['order_id']==sample_id)
    sample = sample.get('features',sample)
    # Reset on every switch, including returning to a previously edited sample.
    if st.session_state.get('last_sample') != sample_id:
        for key in list(st.session_state):
            if key.startswith('input_') or key in ('last_result','last_features','explanation'): del st.session_state[key]
        st.session_state['last_sample'] = sample_id
    left,right=st.columns([3,2])
    with left:
        features=structured_inputs(sample,stats,'input_'+sample_id)
        with st.expander('Advanced: edit raw JSON'):
            use_raw=st.checkbox('Use raw JSON instead of the form',key='input_use_raw')
            raw=st.text_area('Engineered features (JSON)',json.dumps({k:sample[k] for k in stats['features']['numeric']+stats['features']['categorical']},indent=2),height=260,key='input_raw')
        if use_raw:
            try:
                features=json.loads(raw)
                if not isinstance(features,dict): raise ValueError('Features must be a JSON object.')
            except ValueError as exc:
                st.error(str(exc)); features={}
        errors=validate_form(features,stats)
        for error in errors: st.error(error)
        if not errors: range_warnings(features,stats)
        if st.button('Check delivery risk',type='primary',disabled=bool(errors),key='predict_order'):
            try:
                with st.spinner('Checking order with the deployed API…'):
                    result=api_request(api_url,'/predict',{'features':features})
                st.session_state['last_result']=result
                st.session_state['last_features']=features
                st.session_state.pop('explanation',None)
                st.session_state.setdefault('history',[]).append({'order_id':sample_id,'late_probability':result['late_probability']})
            except (ValueError,requests.RequestException) as exc: st.error(str(exc))
    with right:
        with st.container(border=True):
            if st.session_state.get('last_result'):
                result_panel(st.session_state['last_result'],stats,api_url,st.session_state['last_features'])
            else:
                st.subheader('Ready to check')
                st.write('The score and June comparison appear here after submission.')
        with st.expander('What-if explorer'):
            st.caption('Sensitivity scenario only; changing a promise does not establish a causal delivery improvement.')
            feature=st.selectbox('Change one feature',['promised_lead_days','order_frieght_value_sum'],key='what_if_feature')
            limits=stats.get('numeric_ranges',{}).get(feature,{'p01':0,'p99':50})
            bounds=st.slider('Scenario range',float(limits['p01']),float(limits['p99']), (float(limits['p01']),float(limits['p99'])),key='what_if_'+feature)
            if st.button('Compare scenarios',disabled=bool(errors)):
                grid=sorted(set(int(bounds[0]+i*(bounds[1]-bounds[0])/20) if feature=='promised_lead_days' else bounds[0]+i*(bounds[1]-bounds[0])/20 for i in range(21)))
                try:
                    response=api_request(api_url,'/predict/batch',{'rows':[{'features':dict(features,**{feature:value})} for value in grid]})
                    rows=[r.get('prediction',{}) for r in response.get('results',[])]
                    chart=pd.DataFrame({LABELS[feature][0]:grid,'Late probability':[r.get('late_probability') for r in rows]})
                    st.line_chart(chart.set_index(LABELS[feature][0]))
                except (ValueError,requests.RequestException) as exc: st.error(str(exc))
    if st.session_state.get('history'):
        with st.expander('Orders checked this session'):
            frame=pd.DataFrame(st.session_state['history']);st.dataframe(frame)
            st.download_button('Download session history',frame.to_csv(index=False),'session_scores.csv','text/csv')


def replay_page(api_url,stats):
    st.title('Daily watch-list replay')
    st.write('Simulate the operations queue for a June 2018 approval day. Ranking uses model scores only. Outcomes remain hidden until you reveal them.')
    try:
        replay=api_request(api_url,'/replay')
        days=replay.get('available_days',[])
        day=st.selectbox('June approval day',days)
        if not day: st.info('No replay days available.');return
        rows=api_request(api_url,f'/replay?day={day}').get('orders',[])
        if st.button('Score this day',type='primary'):
            all_results=[]
            with st.spinner('Scoring the historical queue…'):
                for start in range(0,len(rows),1000):
                    part=rows[start:start+1000]
                    scored=api_request(api_url,'/predict/batch',{'rows':[{'order_id':r['order_id'],'features':r.get('features',{k:v for k,v in r.items() if k!='order_id'})} for r in part]})
                    for row,result in zip(part,scored['results']): all_results.append(dict(result.get('prediction',{'error':result.get('error')}),order_id=row['order_id'],customer_state=row.get('features',row).get('customer_state')))
            st.session_state['replay_scores']=(day,all_results)
        saved=st.session_state.get('replay_scores')
        if saved and saved[0]==day:
            frame=pd.DataFrame(saved[1])
            if 'late_probability' not in frame: st.error('No valid scores returned.');return
            frame=frame.sort_values('late_probability',ascending=False).reset_index(drop=True)
            frame['rank']=range(1,len(frame)+1);frame['watch_list']=frame['rank']<=math.ceil(len(frame)*0.1)
            st.metric('Orders to watch',int(frame.watch_list.sum()))
            st.caption('Daily top 10% by rank (rounded up); this differs from the fixed June reference score boundary.')
            inspect_id=st.selectbox('Inspect a ranked order',frame.order_id.tolist())
            if st.button('Open selected order'):
                st.session_state['replay_selected']=next(r for r in rows if r['order_id']==inspect_id)
                st.session_state['open_order']=inspect_id
                st.session_state['next_page']='Check an order'
                st.rerun()
            st.dataframe(frame[['rank','order_id','late_probability','watch_list','customer_state']],width='stretch')
            if st.checkbox('Show what actually happened (historical backtest)'):
                outcomes=api_request(api_url,f'/outcomes?day={day}')
                values=outcomes.get('outcomes',{})
                if isinstance(values,list): values={r['order_id']:r.get('is_late',r.get('late')) for r in values}
                frame['actual_late']=frame.order_id.map(values)
                st.warning('Display-only historical outcomes. June labels were never used to fit the model, choose samples, or tune the reference boundary.')
                st.dataframe(frame[['rank','late_probability','watch_list','actual_late','order_id']].assign(actual_late=frame.actual_late.map({0:'On time',1:'Late'}).fillna('Unknown')))
                st.caption(f'{frame.actual_late.isna().sum()} unknown outcomes excluded from historical capture calculations.')
                total=frame.actual_late.fillna(0).sum()
                if total: st.metric('Historical late orders captured',f'{frame.loc[frame.watch_list,"actual_late"].fillna(0).sum()/total:.1%}')
            st.bar_chart(frame.groupby('customer_state').late_probability.mean(),y_label='Mean predicted late probability')
            st.download_button('Download ranked queue',frame.to_csv(index=False),'daily_watch_list.csv','text/csv')
        with st.expander('Upload a batch CSV'):
            st.caption('CSV must contain exactly the 24 engineered feature columns; blank numeric cells become null. Maximum 1,000 rows per upload.')
            template=pd.DataFrame(columns=stats['features']['numeric']+stats['features']['categorical'])
            st.download_button('Download feature template',template.to_csv(index=False),'feature_template.csv','text/csv')
            upload=st.file_uploader('Engineered order features',type=['csv'])
            if upload is not None and st.button('Score uploaded orders'):
                incoming=pd.read_csv(io.BytesIO(upload.getvalue()))
                if len(incoming)>1000: st.error('Upload at most 1,000 orders.');return
                records=json.loads(incoming.to_json(orient='records'))
                response=api_request(api_url,'/predict/batch',{'rows':[{'features':r} for r in records]})
                scored=pd.concat([incoming.reset_index(drop=True),pd.DataFrame([r.get('prediction',{'error':r.get('error')}) for r in response['results']])],axis=1)
                if 'late_probability' in scored: scored['risk_rank']=scored.late_probability.rank(ascending=False,method='min')
                st.dataframe(scored);st.download_button('Download scored CSV',scored.to_csv(index=False),'batch_scores.csv','text/csv')
    except (ValueError,requests.RequestException) as exc: st.error(str(exc))


def performance_page(api_url,stats):
    st.title('Model performance')
    st.info('Historical Phase 2 evaluation. These metrics are not live prediction outcomes.')
    frame=pd.DataFrame(stats['evaluation']).T
    june=stats['evaluation']['June'];cols=st.columns(3)
    cols[0].metric('Historical June AP',f'{june["avg_precision"]:.2%}')
    cols[1].metric('Historical June ROC-AUC',f'{june["roc_auc"]:.3f}')
    cols[2].metric('Historical June top-10% lift',f'{june["top10_lift"]:.2f}×')
    comparison=frame.reindex(['Validation','June','July','August']).reset_index(names='Split')
    for field,label in [('avg_precision','Average precision'),('top10_lift','Top-10% lift (×)')]:
        st.altair_chart(alt.Chart(comparison).mark_bar(color='#E67E22').encode(x=alt.X('Split:N',sort=['Validation','June','July','August']),y=alt.Y(f'{field}:Q',title=label)),width='stretch')
    st.dataframe(frame,width='stretch')
    st.caption(stats.get('evaluation_notes',''))
    importance=stats.get('global_importance',{})
    if importance:
        with st.expander('Global permutation importance'):
            split=st.selectbox('Importance evaluation split',list(importance))
            st.caption('Historical permutation importance; correlated predictors may share importance.')
            st.dataframe(pd.DataFrame(importance[split]))
    st.subheader('Inference monitoring')
    st.caption('In-memory service counters reset on restart. No personal data is stored; this is not realised performance monitoring.')
    st.button('Refresh monitoring')
    try:
        summary=api_request(api_url,'/logs/summary')
        monitoring=st.columns(3)
        monitoring[0].metric('API inference requests',summary.get('request_count',0))
        mean=summary.get('mean_score')
        monitoring[1].metric('Mean recent score',f'{mean:.1%}' if mean is not None else 'No predictions')
        share=summary.get('share_above_watch_list_boundary')
        monitoring[2].metric('Above June reference boundary',f'{share:.1%}' if share is not None else 'No predictions')
        warnings=summary.get('recent_warnings',[])
        if warnings: st.dataframe(pd.DataFrame(warnings),width='stretch')
        st.caption(summary.get('counting_policy','Counters and summaries describe inference activity, not delivery outcomes.'))
    except (ValueError,requests.RequestException) as exc: st.error(str(exc))


def main():
    st.set_page_config(page_title='Delivery Watch · C3',page_icon='📦',layout='wide')
    api_url=os.environ.get('CLASSIFICATION_API_URL','http://127.0.0.1:8002')
    st.sidebar.title('Delivery Watch')
    st.sidebar.caption('Operations & logistics · Olist simulation')
    if st.sidebar.button('Retry API connection'):
        model_info.clear()
    try:
        with st.spinner('Connecting / waking up the API. A sleeping service can take up to a minute…'):
            health=api_request(api_url,'/health')
            stats=model_info(api_url)
        st.sidebar.success('API healthy' if health.get('model_loaded',True) else 'API connected')
    except (ValueError,requests.RequestException) as exc:
        st.sidebar.error('API unavailable');st.error(f'Cannot reach the classification service: {exc}')
        st.info('Start the backend or check CLASSIFICATION_API_URL, then retry. A Render service may be waking up.')
        st.sidebar.caption(api_url);return
    st.sidebar.caption(api_url)
    if st.session_state.get('next_page'):
        st.session_state['workspace']=st.session_state.pop('next_page')
    page=st.sidebar.radio('Workspace',['Check an order','Daily watch-list replay','How to read the score','Model performance','About'],key='workspace')
    if page=='Check an order':
        try:
            data=api_request(api_url,'/samples');samples=data.get('samples',[]) if isinstance(data,dict) else data
            if isinstance(data,dict) and data.get('model_version',stats['model_version'])!=stats['model_version']:
                st.error('API samples and model versions disagree. Refresh the API connection.');return
            extra=st.session_state.get('replay_selected')
            if extra and extra['order_id'] not in [r['order_id'] for r in samples]: samples.append(extra)
            check_order(api_url,stats,samples)
        except (ValueError,requests.RequestException) as exc: st.error(str(exc))
    elif page=='Daily watch-list replay': replay_page(api_url,stats)
    elif page=='How to read the score':
        st.title('How to read the score')
        st.write('The probability estimates the chance an approved order will arrive after its promised date under this historical model. It is a prioritisation signal, not a guarantee or a causal forecast.')
        st.subheader('Rank the queue, then review')
        st.write('The daily replay ranks the whole day and highlights the highest-scoring 10%. This makes review workload explicit. The fixed June top-10% boundary provides historical context for one order; a new day’s own top 10% can differ.')
        st.warning('The fixed 0.5 classification cutoff is illustrative. No historical June score reached 0.5, so June recall at that cutoff was 0%. A below-cutoff order must not be called safe.')
        st.write('Average precision evaluates ranking across cutoffs. Top-10% precision and recall describe a cohort ranking, not a single-order threshold. Later-month results show that performance can change over time.')
        st.write('To use this demo: choose an order → check probability and unusual inputs → review sensitivity drivers → compare a scenario → use daily replay for a queue-level decision.')
    elif page=='Model performance': performance_page(api_url,stats)
    else:
        st.title('About Delivery Watch')
        st.write('A proof of concept for the operations and logistics team deciding which newly approved orders need attention. Simulations use held-out June 2018 Brazilian e-commerce orders.')
        st.write(f'C3 tuned Random Forest · {stats["train_rows"]:,} known-outcome history orders before {stats["training_run_date"]}. The saved preprocessing and fitted forest are served by an independent FastAPI backend.')
        st.warning('2018 data may not represent current operations. Scores are not a service guarantee. New categorical values require retraining; null numerics use training medians. No delivery outcome is supplied as a predictor. June labels are display-only.')
        st.link_button('Project repository and report','https://github.com/mushroomyyy/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/tree/codex/classification-deployment')
        report=os.environ.get('CLASSIFICATION_REPORT_URL')
        if report: st.link_button('Read the project report',report)
        regression=os.environ.get('REGRESSION_APP_URL')
        if regression: st.link_button('Open delay-days regression app',regression)
        with st.expander('Model card and input schema'):st.json({k:stats.get(k) for k in ['model_code','model_version','training_population','best_params','features','categories','export_verification']})
    st.caption(f'{stats["model_code"]} · {stats["model_version"]} · API connected · Historical simulation')


if __name__=='__main__': main()
