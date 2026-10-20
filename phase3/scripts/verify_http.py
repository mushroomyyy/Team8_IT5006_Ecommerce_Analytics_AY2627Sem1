"""Verify a freshly started local or deployed API over HTTP.

Usage: python phase3/scripts/verify_http.py https://actual-service-url
Uses the standard library only. Never records raw input data or credentials.
"""
from __future__ import annotations
import json
import math
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def equivalent(left, right):
    """Allow last-bit BLAS differences while rejecting material output changes."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def verify(base):
    base = base.rstrip('/')
    def call(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(base+path, data=body, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=90) as response:
            return json.load(response)
    start = time.perf_counter()
    health = call('/health')
    metadata = call('/model-info')
    if not call('/samples')['samples']:
        raise RuntimeError('Demonstration samples unavailable')
    # Invented input values: no identifiers or copied transaction records sent.
    features = {
        'order_approved_day_of_week': 5, 'order_approved_day_of_month': 1, 'order_approved_month': 6,
        'order_item_count': 1, 'order_seller_count': 1, 'order_frieght_value_sum': 12.5,
        'order_product_weight_g_sum': 800.0, 'order_product_category_count': 1,
        'seller_dist_km_mean': 350.0, 'seller_state_count': 1, 'payment_value_sum': 100.0,
        'promised_lead_days': 20, 'approval_lag_hours': 2.0, 'order_product_volume_cm3_sum': 1000.0,
        'order_product_photos_mean': 3.0, 'freight_to_price_ratio': 0.12, 'payment_installments_max': 1,
        'is_weekend_approval': 0, 'is_black_friday_period': 0, 'is_december_peak': 0,
        'orders_approved_prev_7d': 1000, 'customer_state': 'SP',
        'route_type': '1. All interstate', 'payment_combination': 'credit_card',
    }
    single = call('/predict', {'features': features})
    batch = call('/predict/batch', {'rows': [{'features': features, 'order_id': 'synthetic-verification'}]})
    if not equivalent(single, batch['results'][0]['prediction']):
        raise RuntimeError('Single/batch predictions differ')
    if 'is_predicted_late' in single['classification'] or 'threshold' in single['classification']:
        raise RuntimeError('Unexpected threshold policy in classification result')
    config = call('/simulation')
    for day in (config['minimum_date'], config['maximum_date']):
        worklist = call('/worklist?simulation_date='+day)
        if 'actual_label' in json.dumps(worklist) or 'delivered_at' in json.dumps(worklist):
            raise RuntimeError('Future outcome/timestamp exposed in operational response')
        counts = worklist['counts']
        if counts['total'] != counts['eligible']+counts['excluded']:
            raise RuntimeError('Worklist count mismatch')
        historical = call('/outcomes?simulation_date='+day)
        if not historical['display_only']:
            raise RuntimeError('Retrospective outcome notice missing')
    for path, payload in [('/worklist?simulation_date=2018-06-01', None), ('/predict', {'features': {}})]:
        try:
            call(path, payload)
        except HTTPError as exc:
            if exc.code != 422: raise
        else:
            raise RuntimeError('Invalid request was not rejected')
    with urlopen(base+'/docs', timeout=90) as docs:
        if docs.status != 200: raise RuntimeError('Swagger unavailable')
    return {'base_url': base, 'health': health, 'checks': ['health', 'model-info', 'samples', 'single-batch numerical equivalence (1e-12 tolerance)',
            'both date boundaries', 'outcome separation', 'retrospective endpoint', 'invalid date/input 422', 'Swagger'],
            'elapsed_seconds': round(time.perf_counter()-start, 3)}

if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:18023'), indent=2))
