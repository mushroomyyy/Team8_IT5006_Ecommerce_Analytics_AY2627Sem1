"""Deterministic daily policy using only knowledge available at the cutoff."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from math import ceil

from .feature_validation import FeatureValidationError


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
MAXIMUM_DATE = date(2018, 7, 1)


def configuration(data):
    return {"minimum_date": str(max(MINIMUM_DATE, snapshot_date(data["model_available_date"]))), "maximum_date": str(MAXIMUM_DATE),
            "default_date": str(max(MINIMUM_DATE, snapshot_date(data["model_available_date"]))), "model_available_date": data["model_available_date"],
            "batch_definition": data["batch_definition"], "timezone": data["timezone"],
            "limitations": data.get("limitations", [])}


def build_worklist(data, simulation_date, review_proportion, minimum_days_remaining, score_rows, validate_row=None):
    minimum_date = max(MINIMUM_DATE, snapshot_date(data.get("model_available_date", str(MINIMUM_DATE))))
    if not minimum_date <= simulation_date <= MAXIMUM_DATE:
        raise ValueError(f"simulation_date must be between {minimum_date} and {MAXIMUM_DATE}")
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
        if validate_row is not None:
            try:
                validate_row(row["features"])
            except FeatureValidationError:
                exclusions.append("invalid_model_inputs")
        if exclusions:
            for reason in exclusions:
                reasons[reason] = reasons.get(reason, 0) + 1
            excluded.append({"order_id": row["order_id"], "promised_delivery_date": row.get("promised_delivery_date"),
                             "exclusion_reasons": exclusions})
        else:
            eligible.append(row)
    predictions = score_rows([row["features"] for row in eligible]) if eligible else []
    if len(predictions) != len(eligible):
        raise SimulationDataError("Prediction count does not match eligible order count")
    ranked = []
    for row, prediction in zip(eligible, predictions):
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
