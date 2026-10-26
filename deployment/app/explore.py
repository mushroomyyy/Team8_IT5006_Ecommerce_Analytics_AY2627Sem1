"""Explore the data: the Phase 1 dashboard rebuilt on Phase 2 definitions, plus model outputs.

Reads only the small tables written by deployment/export/export_eda.py, never the raw data.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

EDA = Path(__file__).resolve().parent / 'data' / 'eda'
FOLLOW, OTHER, INK, MUTED, TEAL = '#E67E22', '#9FB3C8', '#16324F', '#5B7083', '#2A9D8F'
STATUS_LABELS = {'on_time_delivered': 'On time', 'late_delivered': 'Late (delivered)',
                 'late_undelivered': 'Late (not delivered by promise)', 'unresolved': 'Unknown', 'canceled_or_unavailable': 'Canceled or unavailable', 'invalid_or_missing_dates': 'Invalid dates'}
STATES = {'AC': 'Acre', 'AL': 'Alagoas', 'AM': 'Amazonas', 'AP': 'Amapá', 'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal',
          'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão', 'MG': 'Minas Gerais', 'MS': 'Mato Grosso do Sul', 'MT': 'Mato Grosso',
          'PA': 'Pará', 'PB': 'Paraíba', 'PE': 'Pernambuco', 'PI': 'Piauí', 'PR': 'Paraná', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
          'RO': 'Rondônia', 'RR': 'Roraima', 'RS': 'Rio Grande do Sul', 'SC': 'Santa Catarina', 'SE': 'Sergipe', 'SP': 'São Paulo', 'TO': 'Tocantins'}


@st.cache_resource(show_spinner=False)
def load_tables():
    # Shared read-only tables; about 25 MB in memory.
    orders = pd.read_parquet(EDA / 'orders.parquet')
    orders['late'] = orders['late'].astype('float')
    orders['days_from_promise'] = orders['days_from_promise'].astype('float')
    return {'orders': orders, 'sellers': pd.read_parquet(EDA / 'seller_history.parquet'),
            'repeat': pd.read_parquet(EDA / 'repeat_monthly.parquet'),
            'checks': json.loads((EDA / 'data_checks.json').read_text())}


def late_rate(frame, by, min_orders=1):
    """Known-outcome late rate per group, with counts; groups below min_orders dropped."""
    known = frame[frame['late'].notna()]
    table = known.groupby(by, observed=True).agg(orders=('late', 'size'), late_rate=('late', 'mean')).reset_index()
    return table[table['orders'] >= min_orders]


def rate_bars(table, x, title, sort=None, height=300):
    return alt.Chart(table).mark_bar(color=FOLLOW).encode(
        x=alt.X(f'{x}:N', title=None, sort=sort or '-y', axis=alt.Axis(labelAngle=-30, labelLimit=180, labelOverlap=False)),
        y=alt.Y('late_rate:Q', title='Late-delivery rate', axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip(f'{x}:N', title=title), alt.Tooltip('late_rate:Q', title='Late rate', format='.1%'),
                 alt.Tooltip('orders:Q', title='Orders with known outcome', format=',')]).properties(height=height)


def count_bars(table, x, y, title, color=INK, horizontal=False, height=300, fmt=','):
    if horizontal:
        enc = dict(y=alt.Y(f'{x}:N', title=None, sort='-x', axis=alt.Axis(labelOverlap=False, labelLimit=200)), x=alt.X(f'{y}:Q', title=title))
    else:
        enc = dict(x=alt.X(f'{x}:N', title=None, sort='-y', axis=alt.Axis(labelAngle=-30, labelLimit=180)), y=alt.Y(f'{y}:Q', title=title))
    return alt.Chart(table).mark_bar(color=color).encode(**enc, tooltip=[alt.Tooltip(f'{x}:N'), alt.Tooltip(f'{y}:Q', title=title, format=fmt)]).properties(height=height)


def volume_and_rate(table, x, bar, line, bar_title, line_title, line_format='%', height=320):
    base = alt.Chart(table).encode(x=alt.X(f'{x}:T', title=None, axis=alt.Axis(format='%b %Y')))
    bars = base.mark_bar(color=OTHER, opacity=0.8).encode(y=alt.Y(f'{bar}:Q', title=bar_title),
                                                          tooltip=[alt.Tooltip(f'{x}:T', format='%b %Y'), alt.Tooltip(f'{bar}:Q', title=bar_title, format=',')])
    lines = base.mark_line(color=FOLLOW, point=alt.OverlayMarkDef(color=FOLLOW), strokeWidth=2.5).encode(
        y=alt.Y(f'{line}:Q', title=line_title, axis=alt.Axis(format=line_format, titleColor=FOLLOW)),
        tooltip=[alt.Tooltip(f'{x}:T', format='%b %Y'), alt.Tooltip(f'{line}:Q', title=line_title, format='.1%' if line_format == '%' else ',.0f')])
    return alt.layer(bars, lines).resolve_scale(y='independent').properties(height=height)


def card(title, chart, note=None):
    with st.container(border=True):
        st.markdown(f'**{title}**')
        st.altair_chart(chart, width='stretch')
        if note: st.caption(note)


def filters(orders):
    """Filter row shared by the order-based tabs."""
    months = sorted(orders['approved_month'].dropna().unique())
    with st.container(border=True):
        cols = st.columns([2.2, 1.6, 1.6, 1.8])
        start, end = cols[0].select_slider('Approval months', options=months, value=(months[0], months[-1]),
                                           format_func=lambda m: pd.Timestamp(m).strftime('%b %Y'), key='eda_months')
        states = cols[1].multiselect('Customer states', sorted(orders['customer_state'].dropna().unique()), key='eda_states', placeholder='All states')
        routes = cols[2].multiselect('Route type', sorted(orders['route_type'].dropna().unique()), key='eda_routes', placeholder='All routes')
        top = orders['primary_category'].value_counts().index.tolist()
        categories = cols[3].multiselect('Product category', top, key='eda_categories', placeholder='All categories')
    mask = orders['approved_month'].between(start, end)
    if states: mask &= orders['customer_state'].isin(states)
    if routes: mask &= orders['route_type'].isin(routes)
    if categories: mask &= orders['primary_category'].isin(categories)
    return orders[mask]


def overview_tab(orders):
    monthly = orders.groupby('approved_month').agg(orders=('order_id', 'size'), revenue=('payment_value', 'sum')).reset_index()
    monthly['average_order'] = monthly['revenue'] / monthly['orders']
    left, right = st.columns(2)
    with left:
        card('Approved orders and average order value by month',
             volume_and_rate(monthly, 'approved_month', 'orders', 'average_order', 'Approved orders', 'Average order value (R$)', line_format=',.0f'),
             'November 2017 (Black Friday) is the busiest month.')
    with right:
        cats = orders['primary_category'].value_counts().head(12).rename_axis('category').reset_index(name='orders')
        card('Top categories by orders', count_bars(cats, 'category', 'orders', 'Orders', horizontal=True), 'Category of each order’s first item.')
    left, right = st.columns(2)
    with left:
        status = orders['order_status'].value_counts().rename_axis('Status').reset_index(name='Orders')
        status['Share'] = status['Orders'] / status['Orders'].sum()
        with st.container(border=True):
            st.markdown('**Order status**')
            st.dataframe(status, hide_index=True, width='stretch', column_config={'Orders': st.column_config.NumberColumn(format='localized'),
                         'Share': st.column_config.ProgressColumn(format='percent', min_value=0.0, max_value=1.0)})
            st.caption('Approved orders only (payment processed), as in Phase 2.')
    with right:
        heat = orders.groupby(['purchase_weekday', 'purchase_hour']).size().reset_index(name='orders')
        heat['weekday'] = heat['purchase_weekday'].map(dict(enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 1)))
        chart = alt.Chart(heat).mark_rect().encode(
            x=alt.X('purchase_hour:O', title='Hour of purchase'), y=alt.Y('weekday:N', title=None, sort=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
            color=alt.Color('orders:Q', title='Orders', scale=alt.Scale(scheme='oranges')), tooltip=['weekday:N', 'purchase_hour:O', alt.Tooltip('orders:Q', format=',')]).properties(height=300)
        card('When customers buy', chart, 'Purchase time, Brazilian local time.')


def bands(series, edges, labels):
    return pd.cut(series, bins=edges, labels=labels, right=False).astype('string')


def delivery_tab(orders):
    monthly = late_rate(orders, 'approved_month')
    left, right = st.columns(2)
    with left:
        card('Orders and late-delivery rate by month', volume_and_rate(monthly, 'approved_month', 'orders', 'late_rate', 'Orders with known outcome', 'Late-delivery rate'),
             'Late = delivered after the promised day, or still undelivered once it passed (Phase 2 label, checked 45 days after approval).')
    with right:
        days = orders['days_from_promise'].dropna()
        low, high = days.quantile([0.01, 0.99])
        hist = pd.cut(days[days.between(low, high)], bins=range(int(low) - 1, int(high) + 2)).value_counts().sort_index()
        frame = pd.DataFrame({'days': [b.left + 1 for b in hist.index], 'orders': hist.values})
        frame['side'] = frame['days'].gt(0).map({True: 'Late', False: 'On time or early'})
        chart = alt.Chart(frame).mark_bar().encode(
            x=alt.X('days:Q', title='Days from promised date (negative = early)'), y=alt.Y('orders:Q', title='Orders'),
            color=alt.Color('side:N', scale=alt.Scale(domain=['On time or early', 'Late'], range=[OTHER, FOLLOW]), legend=alt.Legend(orient='top', title=None)),
            tooltip=['days:Q', alt.Tooltip('orders:Q', format=',')]).properties(height=320)
        card('How early or late orders arrive', chart, 'The timing model’s target: min(lead days, 45) − promised lead days; canceled orders excluded; 1st–99th percentile shown.')
    left, right = st.columns(2)
    with left:
        outcome = orders['label_status'].map(STATUS_LABELS).value_counts().rename_axis('Outcome').reset_index(name='Orders')
        outcome['Share'] = outcome['Orders'] / outcome['Orders'].sum()
        with st.container(border=True):
            st.markdown('**Delivery outcome against the promise**')
            st.dataframe(outcome, hide_index=True, width='stretch', column_config={'Orders': st.column_config.NumberColumn(format='localized'),
                         'Share': st.column_config.ProgressColumn(format='percent', min_value=0.0, max_value=1.0)})
            st.caption('Unknown = not yet delivered 45 days after approval and the promise had not passed. Unknown and canceled orders are left out of late rates.')
    with right:
        frame = orders.assign(window=bands(orders['promised_lead_days'], [-999, 10, 20, 30, 40, 999], ['< 10 days', '10–19', '20–29', '30–39', '40+']))
        card('Late rate by promised delivery window', rate_bars(late_rate(frame, 'window'), 'window', 'Promised window', sort=['< 10 days', '10–19', '20–29', '30–39', '40+']),
             'Days from approval to the promised date. The shortest promises are missed most often, the longest least.')
    left, right = st.columns(2)
    with left:
        routes = late_rate(orders, 'route_type')
        routes['route_type'] = routes['route_type'].astype(str).str.replace(r'^\d+\.\s*', '', regex=True)
        card('Late rate by route type', rate_bars(routes, 'route_type', 'Route'), 'Whether the sellers are in the customer’s state; Mixed = some are, some are not.')
    with right:
        frame = orders.assign(distance=bands(orders['seller_dist_km_mean'], [0, 100, 300, 600, 1000, 2000, 99999], ['< 100 km', '100–299', '300–599', '600–999', '1,000–1,999', '2,000+']))
        card('Late rate by seller–customer distance', rate_bars(late_rate(frame, 'distance'), 'distance', 'Distance', sort=['< 100 km', '100–299', '300–599', '600–999', '1,000–1,999', '2,000+']),
             'Straight-line distance, averaged over the order’s sellers.')
    delivered = orders[orders['label_status'].isin(['on_time_delivered', 'late_delivered'])].copy()
    delivered['outcome'] = delivered['late'].map({0.0: 'On time', 1.0: 'Late'})
    delivered['carrier_to_customer'] = delivered['approval_to_customer_days'] - delivered['approval_to_carrier_days']
    stages = delivered.groupby('outcome').agg(**{'Approval → carrier': ('approval_to_carrier_days', 'median'), 'Carrier → customer': ('carrier_to_customer', 'median')}).reset_index()
    stages = stages.melt(id_vars='outcome', var_name='stage', value_name='days')
    left, right = st.columns(2)
    with left:
        chart = alt.Chart(stages).mark_bar().encode(
            y=alt.Y('outcome:N', title=None), x=alt.X('days:Q', title='Median days', stack='zero'),
            color=alt.Color('stage:N', scale=alt.Scale(range=[TEAL, OTHER]), legend=alt.Legend(orient='top', title=None)),
            tooltip=['outcome:N', 'stage:N', alt.Tooltip('days:Q', format='.1f')]).properties(height=200)
        card('Where the time goes', chart, 'Delivered orders. Late orders lose most of their time after handover to the carrier.')
    with right.container(border=True):
        st.markdown('**Late rate by order attribute**')
        options = {'Payment method': 'payment_combination', 'Number of items': 'item_count', 'Number of sellers': 'seller_count', 'Instalments': 'payment_installments_max'}
        choice = st.selectbox('Attribute', list(options), key='eda_attribute', label_visibility='collapsed')
        table = late_rate(orders, options[choice], min_orders=100)
        table[options[choice]] = table[options[choice]].astype(str)
        st.altair_chart(rate_bars(table, options[choice], choice, sort='x' if choice != 'Payment method' else '-y', height=220), width='stretch')
        st.caption('Groups with at least 100 orders.')


def geography_tab(orders):
    by_state = late_rate(orders, 'customer_state', min_orders=200)
    by_state['state'] = by_state['customer_state'].astype(str).map(lambda s: f'{s} · {STATES.get(s, s)}')
    left, right = st.columns(2)
    with left:
        card('Late rate by customer state', rate_bars(by_state, 'state', 'State', height=340), 'States with at least 200 orders.')
    with right:
        volume = orders['customer_state'].value_counts().head(10).rename_axis('state').reset_index(name='orders')
        card('Where orders go', count_bars(volume, 'state', 'orders', 'Orders', horizontal=True, height=340), 'Top 10 customer states.')
    routes = orders.dropna(subset=['primary_seller_state', 'customer_state']).assign(
        route=lambda f: f['primary_seller_state'].astype(str) + ' → ' + f['customer_state'].astype(str))
    table = late_rate(routes, 'route', min_orders=500)
    card('Late rate on the busiest routes', rate_bars(table, 'route', 'Seller state → customer state', height=300),
         'Routes with at least 500 orders, using the first item’s seller.')


def sellers_tab(tables, orders):
    history = tables['sellers'][tables['sellers']['order_id'].isin(orders['order_id'])].copy()
    history['experience'] = bands(history['prior_completed_orders'], [0, 1, 10, 50, 200, 10**9], ['New seller', '1–9', '10–49', '50–199', '200+'])
    history['track_record'] = bands(history['prior_late_rate'], [0, 0.05, 0.10, 0.20, 1.01], ['< 5%', '5–10%', '10–20%', '20%+'])
    history['track_record'] = history['track_record'].fillna('No history')
    left, right = st.columns(2)
    with left:
        card('Late rate by seller experience', rate_bars(late_rate(history, 'experience'), 'experience', 'Prior deliveries', sort=['New seller', '1–9', '10–49', '50–199', '200+']),
             'Deliveries the seller had completed before the order was approved.')
    with right:
        card('Late rate by seller track record', rate_bars(late_rate(history, 'track_record'), 'track_record', 'Seller’s past late rate', sort=['No history', '< 5%', '5–10%', '10–20%', '20%+']),
             'Only deliveries completed before approval count, so nothing from the future leaks in.')
    minimum = st.slider('Minimum delivered orders per seller', 5, 200, 30, step=5, key='eda_min_seller')
    per_seller = history.groupby('seller_id').agg(orders=('late', 'size'), late_rate=('late', 'mean')).reset_index()
    per_seller = per_seller[per_seller['orders'] >= minimum]
    chart = alt.Chart(per_seller).mark_circle(color=FOLLOW, opacity=0.6, size=60).encode(
        x=alt.X('orders:Q', title='Delivered orders', scale=alt.Scale(type='log')), y=alt.Y('late_rate:Q', title='Late-delivery rate', axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip('orders:Q', format=','), alt.Tooltip('late_rate:Q', format='.1%')]).properties(height=320)
    card(f'Seller volume and late rate ({len(per_seller):,} sellers)', chart, 'Each dot is a seller; delivered orders only.')


def customers_tab(tables, orders):
    st.caption('Our other two Phase 1 candidate problems: repeat purchases and customer satisfaction.')
    repeat = tables['repeat'].copy()
    repeat['window'] = repeat['complete_window'].map({True: 'Full 3-month window', False: 'Window cut off by data end'})
    chart = alt.Chart(repeat).mark_bar().encode(
        x=alt.X('approved_month:T', title=None, axis=alt.Axis(format='%b %Y')), y=alt.Y('repeat_rate:Q', title='Repeat within 3 months', axis=alt.Axis(format='%')),
        color=alt.Color('window:N', scale=alt.Scale(domain=['Full 3-month window', 'Window cut off by data end'], range=[TEAL, OTHER]), legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('approved_month:T', format='%b %Y'), alt.Tooltip('repeat_rate:Q', format='.2%'), alt.Tooltip('customers:Q', format=',')]).properties(height=300)
    card('Customers who order again within three months', chart, 'Only about 1% of customers return within three months: too rare to predict well, one reason we chose late delivery.')
    left, right = st.columns(2)
    with left:
        scores = orders['review_score'].dropna().round().astype(int).value_counts().sort_index().rename_axis('score').reset_index(name='orders')
        card('Review scores', count_bars(scores.assign(score=scores['score'].astype(str)), 'score', 'orders', 'Orders', color=INK).encode(x=alt.X('score:N', title='Review score', sort=None)))
    with right:
        card('Late rate by review score', rate_bars(late_rate(orders, 'review_group'), 'review_group', 'Review group', sort=['Low (1-2)', 'Mid (3-4)', 'High (5)']),
             'Phase 2 late label, which also counts orders never delivered by the promise. On delivered orders only (the Phase 1 definition) the low-score rate is lower.')


def model_tab(orders, scores, codes):
    risk, timing = codes
    replay = orders[orders['approved_month'] >= pd.Timestamp('2018-06-01')].copy()
    pairs = replay['order_id'].map(scores)
    replay = replay[pairs.notna()]
    pairs = pairs.dropna()
    replay['risk'] = [p[0] for p in pairs]
    replay['timing'] = [p[1] for p in pairs]
    if replay.empty:
        st.info('No June–August 2018 orders match the current filters.'); return
    st.caption(f'June–August 2018, scored with the deployed {risk} and {timing} models. June was the held-out test month; July and August are monitoring months the models never saw.')
    monthly = replay.groupby('approved_month').agg(late_rate=('late', 'mean'), risk=('risk', 'mean'), actual=('days_from_promise', 'mean'), timing=('timing', 'mean')).reset_index()
    left, right = st.columns(2)
    with left:
        long = monthly.melt(id_vars='approved_month', value_vars=['late_rate', 'risk'], var_name='series', value_name='value')
        long['series'] = long['series'].map({'late_rate': 'Actual late rate', 'risk': f'Average {risk} risk'})
        chart = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X('approved_month:T', title=None, axis=alt.Axis(format='%b %Y', tickCount='month')), y=alt.Y('value:Q', title=None, axis=alt.Axis(format='%')),
            color=alt.Color('series:N', scale=alt.Scale(range=[INK, FOLLOW]), legend=alt.Legend(orient='top', title=None)),
            tooltip=[alt.Tooltip('approved_month:T', format='%b %Y'), 'series:N', alt.Tooltip('value:Q', format='.1%')]).properties(height=280)
        card(f'Actual late rate and {risk}’s average risk', chart, f'{risk} scores rank orders and are not calibrated, so compare the trends, not the levels.')
    with right:
        long = monthly.melt(id_vars='approved_month', value_vars=['actual', 'timing'], var_name='series', value_name='days')
        long['series'] = long['series'].map({'actual': 'Actual', 'timing': f'{timing} prediction'})
        chart = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X('approved_month:T', title=None, axis=alt.Axis(format='%b %Y', tickCount='month')), y=alt.Y('days:Q', title='Average days from promise'),
            color=alt.Color('series:N', scale=alt.Scale(range=[INK, FOLLOW]), legend=alt.Legend(orient='top', title=None)),
            tooltip=[alt.Tooltip('approved_month:T', format='%b %Y'), 'series:N', alt.Tooltip('days:Q', format='+.1f')]).properties(height=280)
        card(f'{timing} against the actual buffer', chart, 'Negative = before the promised date.')
    known = replay[replay['late'].notna()].copy()
    known['decile'] = known.groupby('approved_month')['risk'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False) + 1)
    lift = known.groupby(['approved_month', 'decile']).agg(late_rate=('late', 'mean')).reset_index()
    lift['month'] = lift['approved_month'].dt.strftime('%B')
    chart = alt.Chart(lift).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('decile:O', title=f'{risk} risk decile (10 = riskiest tenth of the month)', axis=alt.Axis(labelAngle=0)), y=alt.Y('late_rate:Q', title='Late-delivery rate', axis=alt.Axis(format='%')),
        color=alt.Color('month:N', scale=alt.Scale(domain=['June', 'July', 'August'], range=[FOLLOW, TEAL, INK]), legend=alt.Legend(orient='top', title=None)),
        tooltip=['month:N', 'decile:O', alt.Tooltip('late_rate:Q', format='.1%')]).properties(height=300)
    card(f'Does higher {risk} risk mean more late orders?', chart, 'A useful score slopes upward: the riskiest decile should have the highest late rate. Compare how the slope changes from June to August.')
    by_state = known.groupby('customer_state', observed=True).agg(orders=('late', 'size'), late_rate=('late', 'mean'), risk=('risk', 'mean')).reset_index()
    by_state = by_state[by_state['orders'] >= 100]
    chart = alt.Chart(by_state).mark_circle(color=FOLLOW, size=120, opacity=0.8).encode(
        x=alt.X('risk:Q', title=f'Average {risk} risk', axis=alt.Axis(format='%'), scale=alt.Scale(zero=False, padding=20)), y=alt.Y('late_rate:Q', title='Actual late rate', axis=alt.Axis(format='%')),
        tooltip=['customer_state:N', alt.Tooltip('orders:Q', format=','), alt.Tooltip('late_rate:Q', format='.1%'), alt.Tooltip('risk:Q', format='.1%')])
    labels = chart.mark_text(dx=12, align='left', color=INK, fontSize=12).encode(text='customer_state:N')
    card('Do the model’s risky states match reality?', alt.layer(chart, labels).properties(height=320), 'Customer states with at least 100 orders, June–August 2018.')


def data_tab(tables, orders):
    checks = tables['checks']
    coverage = pd.DataFrame(checks['coverage'])
    coverage['month'] = pd.to_datetime(coverage['month'])
    coverage['in_scope'] = coverage['month'].between(pd.Timestamp(checks['window']['first_month']), pd.Timestamp(checks['window']['last_month']))
    chart = alt.Chart(coverage).mark_bar().encode(
        x=alt.X('month:T', title=None, axis=alt.Axis(format='%b %Y')), y=alt.Y('orders:Q', title='Orders placed'),
        color=alt.Color('in_scope:N', scale=alt.Scale(domain=[True, False], range=[FOLLOW, OTHER]), legend=alt.Legend(orient='top', title='In analysis window', labelExpr="datum.value ? 'Yes' : 'No'")),
        tooltip=[alt.Tooltip('month:T', format='%b %Y'), alt.Tooltip('orders:Q', format=','), alt.Tooltip('approved:Q', format=','), alt.Tooltip('delivered:Q', format=',')]).properties(height=280)
    card('Why the analysis covers January 2017 to August 2018', chart, 'Months outside the window have too few orders or incomplete outcomes.')
    left, right = st.columns(2)
    with left:
        st.markdown('**Source tables**')
        st.dataframe(pd.DataFrame(checks['inventory']).rename(columns={'table': 'Table', 'rows': 'Rows', 'columns': 'Columns', 'missing_cells_pct': 'Missing cells (%)'}), hide_index=True, width='stretch')
    with right:
        st.markdown('**Join and quality checks**')
        st.dataframe(pd.DataFrame(checks['checks']).rename(columns={'check': 'Check', 'result': 'Result'}).astype(str), hide_index=True, width='stretch')
    with st.expander('Definitions (from the Phase 2 report)', expanded=True):
        st.markdown(
            '- **Orders:** approved orders (payment processed), approved January 2017 to August 2018; one row per order.\n'
            '- **Time axis:** approval date (calendar dates).\n'
            '- **Late:** delivered after the promised day, or still undelivered once it passed; checked 45 days after approval. Otherwise unknown and left out of rates. Canceled and unavailable orders have no outcome and are left out too.\n'
            '- **Days from promise:** min(lead days, 45) − promised lead days, lead days from approval; canceled and unavailable orders excluded.\n'
            '- **Distance, route type, payment method:** the Phase 2 engineered features.')
    st.caption(f'{len(orders):,} orders match the current filters.')


def explore_page(page_header, scores, codes):
    page_header('Explore the data', 'Olist orders approved January 2017 to August 2018: what drives late deliveries, plus our two other candidate problems.')
    tables = load_tables()
    orders = filters(tables['orders'])
    if orders.empty:
        st.info('No orders match these filters.'); return
    known = orders['late'].notna()
    cards = st.columns(4)
    cards[0].metric('Approved orders', f'{len(orders):,}', border=True)
    cards[1].metric('Late-delivery rate', f"{orders.loc[known, 'late'].mean():.1%}", help='Share of orders with a known outcome that were late (Phase 2 label).', border=True)
    cards[2].metric('Median days from promise', f"{orders['days_from_promise'].median():+.0f}", help='Negative = arrived before the promised date.', border=True)
    cards[3].metric('Average review score', f"{orders['review_score'].mean():.2f} / 5", border=True)
    tabs = st.tabs(['Overview', 'Delivery', 'Geography', 'Sellers', 'Customers', 'Model outputs', 'Data and definitions'])
    with tabs[0]: overview_tab(orders)
    with tabs[1]: delivery_tab(orders)
    with tabs[2]: geography_tab(orders)
    with tabs[3]: sellers_tab(tables, orders)
    with tabs[4]: customers_tab(tables, orders)
    with tabs[5]: model_tab(orders, scores, codes)
    with tabs[6]: data_tab(tables, orders)
