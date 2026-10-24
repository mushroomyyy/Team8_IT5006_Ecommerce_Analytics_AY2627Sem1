"""Historical demo orchestration; fitted preprocessing and inference stay remote."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import threading
import time

import requests
import streamlit as st

from .simulation import SimulationDataError, build_worklist, retrospective, scoring_candidates

DATA = Path(__file__).resolve().parent / 'data'
# Short per-call budgets: Streamlit cannot interrupt a request in flight, so no
# single call may hold a run (or a refreshed page) for minutes.
HEALTH_TIMEOUT = (5, 10)
REQUEST_TIMEOUT = (5, 60)
WAKE_BUDGET_SECONDS = 120
WAKE_RETRY_SECONDS = 3
BATCH_CHUNK = 100
SCORE_CACHE_ENTRIES = 64


def read_demo_json(name):
    """Read a named packaged input. Outcomes have their own explicit-reveal loader."""
    try:
        return json.loads((DATA / name).read_text())
    except (OSError, ValueError) as exc:
        raise SimulationDataError(f'Historical demonstration data unavailable: {name}') from exc


@st.cache_resource(show_spinner=False, max_entries=2)
def _orders_snapshot(revision):
    # Shared read-only object: parsed once per file revision, never mutated by callers.
    return read_demo_json('simulation_orders.json')


def load_demo_orders():
    return _orders_snapshot(snapshot_revision())


def load_demo_samples():
    return read_demo_json('classification_sample_orders.json')


def reveal_outcomes(worklist):
    # Deliberately never called by input loading, scoring or worklist building.
    outcomes = read_demo_json('simulation_outcomes.json')['outcomes']
    return retrospective(worklist, outcomes)


def load_month_inputs():
    """Pre-scored replay inputs and eventual outcomes, read only by the month-end review page."""
    scored = read_demo_json('replay_scores.json')
    return (scored, read_demo_json('simulation_outcomes.json')['outcomes'],
            read_demo_json('simulation_timing.json')['actual_days_from_promise'])


def snapshot_revision():
    try:
        stat = (DATA / 'simulation_orders.json').stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        raise SimulationDataError('Historical order snapshot unavailable') from exc


def api_request(api_url, endpoint, payload=None, timeout=REQUEST_TIMEOUT):
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.get(url, timeout=timeout) if payload is None else requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f'API returned {response.status_code}: {detail}')
    return response.json()


def wait_for_service(api_url, on_wait=None):
    """Poll /health with short calls until it answers; on_wait(elapsed, error) reports progress between attempts."""
    started = time.monotonic()
    while True:
        try:
            return api_request(api_url, '/health', timeout=HEALTH_TIMEOUT)
        except (ValueError, requests.RequestException) as exc:
            elapsed = time.monotonic() - started
            if elapsed + WAKE_RETRY_SECONDS >= WAKE_BUDGET_SECONDS:
                raise requests.Timeout(f'Prediction service did not respond within {WAKE_BUDGET_SECONDS} seconds ({exc})') from exc
            if on_wait: on_wait(elapsed, exc)
            time.sleep(WAKE_RETRY_SECONDS)


def score_demo_rows(api_url, rows, on_progress=None):
    """Validate row alignment; only explicit API 422 rows become exclusions."""
    predictions = []
    if on_progress and rows: on_progress(0, len(rows))
    for start in range(0, len(rows), BATCH_CHUNK):
        batch = rows[start:start + BATCH_CHUNK]
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
        if on_progress: on_progress(len(predictions), len(rows))
    return predictions


def model_versions(metadata):
    return tuple((name, metadata[name]['model_version']) for name in ('classification', 'regression'))


@st.cache_resource(show_spinner=False)
def _score_store():
    # Process-wide, lock-free score cache. Unlike st.cache_data it never makes a
    # refreshed page wait on another session's request; at worst two sessions
    # score the same day once each.
    return OrderedDict(), threading.Lock()


def clear_caches():
    store, lock = _score_store()
    with lock: store.clear()
    _orders_snapshot.clear()


def day_scores(api_url, versions, simulation_date, revision, on_progress=None):
    """Scores for every row any eligibility setting could include; ranking and selection stay local."""
    key = (api_url, versions, str(simulation_date), revision)
    store, lock = _score_store()
    with lock:
        if key in store:
            store.move_to_end(key)
            return store[key]
    # Superset of eligible rows for every minimum-days setting; scores use
    # approval-time features only, so the cutoff rules are unchanged.
    candidates = scoring_candidates(load_demo_orders(), simulation_date)
    ids = [row['order_id'] for row in candidates]
    if len(set(ids)) != len(ids):
        raise SimulationDataError('Historical batch contains duplicate order IDs')
    predictions = score_demo_rows(api_url, [{'order_id': row['order_id'], 'features': row['features']} for row in candidates], on_progress)
    scores = dict(zip(ids, predictions))
    with lock:
        store[key] = scores
        while len(store) > SCORE_CACHE_ENTRIES: store.popitem(last=False)
    return scores


def build_day_worklist(api_url, versions, simulation_date, review_proportion, minimum_days_remaining, revision, on_progress=None):
    """Score once per day; changing the follow-up proportion or minimum days re-ranks locally without API calls."""
    scores = day_scores(api_url, versions, simulation_date, revision, on_progress)
    return build_worklist(load_demo_orders(), simulation_date, review_proportion, minimum_days_remaining,
                          lambda rows: [scores[row['order_id']] for row in rows])
