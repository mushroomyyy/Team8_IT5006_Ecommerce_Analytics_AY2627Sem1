"""Deterministic daily policy using only knowledge available at the cutoff."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from math import ceil


class SimulationDataError(RuntimeError):
    """Packaged snapshot is malformed or internally inconsistent."""


def snapshot_date(value):
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError) as exc:
        raise SimulationDataError("Malformed historical snapshot date") from exc


def snapshot_datetime(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            raise ValueError("Source convention requires naive local dates")
        return parsed
    except (TypeError, ValueError) as exc:
        raise SimulationDataError("Malformed historical snapshot timestamp") from exc

MINIMUM_DATE = date(2018, 6, 2)
# June guided model selection; July and August are frozen-model monitoring months.
MAXIMUM_DATE = date(2018, 8, 30)
REPLAY_MONTHS = {"2018-06": "June 2018", "2018-07": "July 2018", "2018-08": "August 2018"}


def configuration(data):
    return {"minimum_date": str(max(MINIMUM_DATE, snapshot_date(data["model_available_date"]))), "maximum_date": str(MAXIMUM_DATE),
            "default_date": str(max(MINIMUM_DATE, snapshot_date(data["model_available_date"]))), "model_available_date": data["model_available_date"],
            "batch_definition": data["batch_definition"], "timezone": data["timezone"],
            "limitations": data.get("limitations", [])}


def check_simulation_date(data, simulation_date):
    minimum_date = max(MINIMUM_DATE, snapshot_date(data.get("model_available_date", str(MINIMUM_DATE))))
    if not minimum_date <= simulation_date <= MAXIMUM_DATE:
        raise ValueError(f"simulation_date must be between {minimum_date} and {MAXIMUM_DATE}")


def scoring_candidates(data, simulation_date):
    """Rows any minimum-days setting could make eligible: undelivered at the cutoff and with a promise date."""
    check_simulation_date(data, simulation_date)
    cutoff = datetime.combine(simulation_date, datetime.min.time())
    rows = data["days"].get(str(simulation_date - timedelta(days=1)), [])
    return [row for row in rows if row.get("promised_delivery_date") and
            not (row.get("delivered_at") and snapshot_datetime(row["delivered_at"]) < cutoff)]


def build_worklist(data, simulation_date, review_proportion, minimum_days_remaining, score_rows):
    check_simulation_date(data, simulation_date)
    if isinstance(review_proportion, bool) or not isinstance(review_proportion, (int, float)) or not 0 <= review_proportion <= 1:
        raise ValueError("review_proportion must be between 0 and 1")
    if isinstance(minimum_days_remaining, bool) or not isinstance(minimum_days_remaining, int) or not 0 <= minimum_days_remaining <= 365:
        raise ValueError("minimum_days_remaining must be an integer between 0 and 365")
    cutoff = datetime.combine(simulation_date, datetime.min.time())
    batch_date = simulation_date - timedelta(days=1)
    eligible, excluded, reasons = [], [], {}
    rows = data["days"].get(str(batch_date), [])
    for row in rows:
        exclusions = []
        delivered = snapshot_datetime(row["delivered_at"]) if row.get("delivered_at") else None
        if delivered is not None and delivered < cutoff:
            exclusions.append("already_delivered")
        promise = snapshot_date(row["promised_delivery_date"]) if row.get("promised_delivery_date") else None
        if promise is None:
            exclusions.append("missing_promised_delivery_date")
        elif (promise - simulation_date).days <= minimum_days_remaining:
            exclusions.append("insufficient_time_remaining")
        if exclusions:
            for reason in exclusions:
                reasons[reason] = reasons.get(reason, 0) + 1
            excluded.append({"order_id": row["order_id"], "promised_delivery_date": row.get("promised_delivery_date"),
                             "exclusion_reasons": exclusions})
        else:
            eligible.append(row)
    predictions = score_rows([{ "order_id": row["order_id"], "features": row["features"]} for row in eligible]) if eligible else []
    if len(predictions) != len(eligible):
        raise SimulationDataError("Prediction count does not match eligible order count")
    ranked = []
    for row, prediction in zip(eligible, predictions):
        if prediction is None:
            excluded.append({"order_id": row["order_id"], "promised_delivery_date": row.get("promised_delivery_date"),
                             "exclusion_reasons": ["invalid_model_inputs"]})
            reasons["invalid_model_inputs"] = reasons.get("invalid_model_inputs", 0) + 1
            continue
        ranked.append({"order_id": row["order_id"], "promised_delivery_date": row["promised_delivery_date"],
                       "features": row["features"], **prediction,
                       "late_probability": prediction["classification"]["late_probability"],
                       "predicted_days_from_promise": prediction["regression"]["predicted_days_from_promise"]})
    ranked.sort(key=lambda row: (-row["late_probability"], row["order_id"]))
    selected = ceil(review_proportion * len(ranked))
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, follow_up_recommended=rank <= selected,
                   recommendation="Follow-up recommended" if rank <= selected else "Not selected for follow-up")
    return {"simulation_date": str(simulation_date), "cutoff": cutoff.isoformat(), "batch_date": str(batch_date),
            "timezone": data["timezone"], "batch_definition": data["batch_definition"],
            "review_proportion": review_proportion, "minimum_days_remaining": minimum_days_remaining,
            "counts": {"total": len(rows), "eligible": len(ranked), "excluded": len(excluded), "selected": selected},
            "exclusion_reasons": reasons, "orders": ranked, "excluded_orders": excluded,
            "policy_notice": "Daily proportion is a demo operational setting; not selected does not guarantee on-time delivery."}


def scored_lookup(scores):
    """Adapt pre-scored [late_probability, days_from_promise] pairs to the API prediction shape."""
    def score_rows(rows):
        predictions = []
        for row in rows:
            if row["order_id"] not in scores:
                raise SimulationDataError("Pre-scored June file does not cover this order; rerun phase3/scripts/score_june.py")
            pair = scores[row["order_id"]]
            predictions.append(None if pair is None else {"classification": {"late_probability": pair[0]},
                                                          "regression": {"predicted_days_from_promise": pair[1]}})
        return predictions
    return score_rows


def month_replay(data, scores, known_outcomes, minimum_days_remaining, month=None):
    """Replay every supported day once (optionally one approval month); outcomes attach only after ranking."""
    first = max(MINIMUM_DATE, snapshot_date(data.get("model_available_date", str(MINIMUM_DATE))))
    days = []
    for offset in range((MAXIMUM_DATE - first).days + 1):
        current = first + timedelta(days=offset)
        if month and not str(current - timedelta(days=1)).startswith(month):
            continue
        worklist = build_worklist(data, current, 1, minimum_days_remaining, scored_lookup(scores))
        days.append({"simulation_date": str(current),
                     "ranked": [known_outcomes.get(row["order_id"]) for row in worklist["orders"]],
                     "excluded": [known_outcomes.get(row["order_id"]) for row in worklist["excluded_orders"]]})
    return days


def month_summary(replay, review_proportion):
    """Daily and pooled capture for one follow-up proportion, rounding up each day like the worklist."""
    def late(values): return sum(value == 1 for value in values)
    def known(values): return sum(value in (0, 1) for value in values)
    days, pooled = [], dict.fromkeys(["total", "eligible", "selected", "selected_known", "late_caught", "late_eligible", "late_all"], 0)
    for day in replay:
        ranked = day["ranked"]
        selected = ranked[:ceil(review_proportion * len(ranked))]
        row = {"simulation_date": day["simulation_date"], "total": len(ranked) + len(day["excluded"]),
               "eligible": len(ranked), "selected": len(selected), "selected_known": known(selected),
               "late_caught": late(selected), "late_eligible": late(ranked), "late_all": late(ranked) + late(day["excluded"])}
        days.append(row)
        for key in pooled: pooled[key] += row[key]
    def ratio(numerator, denominator): return numerator / denominator if denominator else None
    pooled.update(selected_late_rate=ratio(pooled["late_caught"], pooled["selected_known"]),
                  eligible_late_coverage=ratio(pooled["late_caught"], pooled["late_eligible"]),
                  all_late_coverage=ratio(pooled["late_caught"], pooled["late_all"]))
    return {"review_proportion": review_proportion, "days": days, "pooled": pooled}


def monthly_ranking(data, scores, known_outcomes, review_proportion, month=None):
    """Phase 2 style comparison: rank a whole approval month at once (needs month-end hindsight of the cohort)."""
    rows = [row for key, day in data["days"].items() if not month or key.startswith(month)
            for row in day if scores.get(row["order_id"]) is not None]
    rows.sort(key=lambda row: (-scores[row["order_id"]][0], row["order_id"]))
    outcomes = [known_outcomes.get(row["order_id"]) for row in rows]
    selected = outcomes[:ceil(review_proportion * len(outcomes))]
    late_all = sum(value == 1 for value in outcomes)
    caught = sum(value == 1 for value in selected)
    return {"total": len(outcomes), "selected": len(selected), "late_caught": caught, "late_all": late_all,
            "all_late_coverage": caught / late_all if late_all else None}


def buffer_ranking_scores(scores):
    """Comparison ranking by R1: the later R1 expects arrival relative to the promise, the riskier."""
    return {key: None if pair is None else [pair[1], pair[1]] for key, pair in scores.items()}


def timing_accuracy(data, scores, actual_days, known_outcomes, month=None):
    """R1 error against its Phase 2 target (days from promise), overall and per approval day."""
    pairs, by_day = [], {}
    for key, day in data["days"].items():
        if month and not key.startswith(month):
            continue
        for row in day:
            pair, actual = scores.get(row["order_id"]), actual_days.get(row["order_id"])
            if pair is None or actual is None:
                continue
            pairs.append((pair[1], actual, known_outcomes.get(row["order_id"])))
            by_day.setdefault(key, []).append((pair[1], actual))
    if not pairs:
        return None
    errors = [predicted - actual for predicted, actual, _ in pairs]
    def mean(values): return sum(values) / len(values) if values else None
    return {"n": len(pairs), "mae": mean([abs(e) for e in errors]), "rmse": mean([e * e for e in errors]) ** 0.5,
            "bias": mean(errors), "mean_actual": mean([a for _, a, _ in pairs]), "mean_predicted": mean([p for p, _, _ in pairs]),
            "predicted_late_orders": mean([p for p, _, o in pairs if o == 1]),
            "predicted_on_time_orders": mean([p for p, _, o in pairs if o == 0]),
            "days": [{"approval_date": key, "actual": mean([a for _, a in values]), "predicted": mean([p for p, _ in values]), "n": len(values)}
                     for key, values in sorted(by_day.items())]}


def retrospective(worklist, known_outcomes):
    eligible_ids = [row["order_id"] for row in worklist["orders"]]
    selected_ids = [row["order_id"] for row in worklist["orders"] if row["follow_up_recommended"]]
    all_ids = eligible_ids + [row["order_id"] for row in worklist["excluded_orders"]]
    outcomes = {key: known_outcomes.get(key) for key in all_ids}
    def counts(ids):
        values = [outcomes[key] for key in ids]
        known = sum(value in (0, 1) for value in values)
        return {"total": len(values), "known": known, "unknown": len(values) - known,
                "late": sum(value == 1 for value in values)}
    all_counts, eligible_counts, selected_counts = map(counts, (all_ids, eligible_ids, selected_ids))
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    metrics = {"all": all_counts, "eligible": eligible_counts, "selected": selected_counts,
               "selected_late_rate": ratio(selected_counts["late"], selected_counts["known"]),
               "eligible_late_coverage": ratio(selected_counts["late"], eligible_counts["late"]),
               "all_late_coverage": ratio(selected_counts["late"], all_counts["late"])}
    return {"simulation_date": worklist["simulation_date"], "display_only": True, "outcomes": outcomes,
            "metrics": metrics, "notice": "Retrospective eventual outcomes are outside simulated cutoff knowledge. Unknown outcomes are excluded from rate denominators. Daily policy results differ from Phase 2 monthly rankings."}
