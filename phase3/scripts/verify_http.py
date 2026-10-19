"""Verify a freshly started local or deployed API over HTTP.

Usage: python phase3/scripts/verify_http.py https://actual-service-url
Uses the standard library only. Never records raw input data or credentials.
"""
from __future__ import annotations
import json
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
    sample = call('/samples')['samples'][0]
    features = {key: value for key, value in sample.items() if key != 'order_id'}
    single = call('/predict', {'features': features})
    batch = call('/predict/batch', {'rows': [{'features': features, 'order_id': sample['order_id']}]})
    if single != batch['results'][0]['prediction']:
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
    return {'base_url': base, 'health': health, 'checks': ['health', 'model-info', 'samples', 'single-batch exact equality',
            'both date boundaries', 'outcome separation', 'retrospective endpoint', 'invalid date/input 422', 'Swagger'],
            'elapsed_seconds': round(time.perf_counter()-start, 3)}

if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:18023'), indent=2))
