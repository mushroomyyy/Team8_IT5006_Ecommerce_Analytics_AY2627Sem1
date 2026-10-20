"""Historical demo orchestration; fitted preprocessing and inference stay remote."""
from __future__ import annotations

import json
import math
from pathlib import Path

import requests
import streamlit as st

from .simulation import SimulationDataError, build_worklist, retrospective

DATA = Path(__file__).resolve().parent / 'data'


def read_demo_json(name):
    """Read a named packaged input. Outcomes have their own explicit-reveal loader."""
    try:
        return json.loads((DATA / name).read_text())
    except (OSError, ValueError) as exc:
        raise SimulationDataError(f'Historical demonstration data unavailable: {name}') from exc


def load_demo_orders():
    return read_demo_json('simulation_orders.json')


def load_demo_samples():
    return read_demo_json('classification_sample_orders.json')


def reveal_outcomes(worklist):
    # Deliberately never called by input loading, scoring or cached_worklist.
    outcomes = read_demo_json('simulation_outcomes.json')['outcomes']
    return retrospective(worklist, outcomes)


def snapshot_revision():
    try:
        stat = (DATA / 'simulation_orders.json').stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        raise SimulationDataError('Historical order snapshot unavailable') from exc


def api_request(api_url, endpoint, payload=None):
    response = (requests.get(f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}", timeout=(10, 120))
                if payload is None else requests.post(f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}", json=payload, timeout=(10, 120)))
    if not response.ok:
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f'API returned {response.status_code}: {detail}')
    return response.json()


def score_demo_rows(api_url, rows):
    """Validate row alignment; only explicit API 422 rows become exclusions."""
    predictions = []
    for start in range(0, len(rows), 1000):
        batch = rows[start:start + 1000]
        response = api_request(api_url, '/predict/batch', {'rows': batch})
        results = response.get('results') if isinstance(response, dict) else None
        if not isinstance(results, list) or len(results) != len(batch):
            raise SimulationDataError('API result count does not match submitted order count')
        for index, (submitted, result) in enumerate(zip(batch, results)):
            if not isinstance(result, dict):
                raise SimulationDataError('API returned an invalid batch result')
            if result.get('index') != index or result.get('order_id') != submitted['order_id']:
                raise SimulationDataError('API row identity does not match submitted order')
            if 'error' in result:
                if result.get('status_code') == 422:
                    predictions.append(None)
                    continue
                raise SimulationDataError(f"API inference failed for a batch row ({result.get('status_code', 'unknown status')}); retry the worklist")
            prediction = result.get('prediction')
            try:
                probability = prediction['classification']['late_probability']
                days = prediction['regression']['predicted_days_from_promise']
                valid = (not isinstance(probability, bool) and isinstance(probability, (int, float)) and
                         math.isfinite(probability) and 0 <= probability <= 1 and
                         not isinstance(days, bool) and isinstance(days, (int, float)) and math.isfinite(days))
                if not valid: raise ValueError('Nonfinite prediction')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise SimulationDataError('API returned an invalid integrated prediction') from exc
            predictions.append(prediction)
    return predictions


def model_versions(metadata):
    return tuple((name, metadata[name]['model_version']) for name in ('classification', 'regression'))


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def cached_worklist(api_url, versions, simulation_date, review_proportion, minimum_days_remaining, revision):
    """Cache only scored operational inputs keyed by service/models/date/policy/snapshot."""
    return build_worklist(load_demo_orders(), simulation_date, review_proportion, minimum_days_remaining,
                          lambda rows: score_demo_rows(api_url, rows))
