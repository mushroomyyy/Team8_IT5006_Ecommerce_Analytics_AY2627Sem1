"""HTTP client for the independently hosted classification API."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def api_request(api_url, endpoint, payload=None):
    """Surface backend validation details instead of hiding them as connection errors."""
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    response = (requests.get(url, timeout=60) if payload is None
                else requests.post(url, json=payload, timeout=60))
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"API returned {response.status_code}: {detail}")
    return response.json()


def main():
    st.set_page_config(page_title="Delivery Risk", page_icon="📦", layout="wide")
    st.title("Delivery risk check")
    st.caption("C3 · Tuned Random Forest · Predictions at order approval")
    api_url = os.environ.get("CLASSIFICATION_API_URL", "http://127.0.0.1:8002")
    st.sidebar.caption(f"Classification API: {api_url}")
    st.sidebar.markdown("Use held-out June 2018 orders to simulate incoming orders. "
                        "A low score does not guarantee an on-time delivery.")
    try:
        stats = api_request(api_url, "/model-info")
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Cannot load the classification API: {exc}")
        st.info("Start the backend or set CLASSIFICATION_API_URL to its deployed HTTPS URL.")
        return

    prediction_tab, stats_tab = st.tabs(["Order prediction", "Model performance"])
    with prediction_tab:
        samples = json.loads((ARTIFACTS / "classification_sample_orders.json").read_text())
        sample_id = st.selectbox("Held-out historical order", [row["order_id"] for row in samples])
        sample = next(row for row in samples if row["order_id"] == sample_id)
        st.write("Review or edit the engineered approval-time features before submitting. "
                 "Null values use the fitted pipeline's missing-value handling. "
                 "Feature calculations must follow the Phase 2 definitions.")
        # A key per order ensures selecting another sample resets the editor correctly.
        with st.form(f"order_{sample_id}"):
            raw = st.text_area("Order features (JSON)",
                               json.dumps({k: v for k, v in sample.items() if k != "order_id"}, indent=2),
                               height=430, key=f"features_{sample_id}")
            submit = st.form_submit_button("Check delivery risk", type="primary")
        if submit:
            try:
                features = json.loads(raw)
                if not isinstance(features, dict):
                    raise ValueError("Features must be a JSON object")
                with st.spinner("Checking order…"):
                    result = api_request(api_url, "/predict", {"features": features})
                st.metric("Estimated late-delivery probability", f"{result['late_probability']:.1%}")
                st.write("Above demo cutoff: review" if result["is_predicted_late"]
                         else "Below demo cutoff: continue monitoring")
                st.info(result["threshold_policy"])
                st.caption(f"Model {result['model_code']} · {result['model_version']}")
                with st.expander("API response"):
                    st.json(result)
            except (ValueError, requests.RequestException) as exc:
                st.error(str(exc))

    with stats_tab:
        st.info("Historical Phase 2 evaluation; these are not live prediction outcomes.")
        june = stats["evaluation"]["June"]
        cols = st.columns(3)
        cols[0].metric("June AP", f"{june['avg_precision']:.2%}")
        cols[1].metric("June ROC-AUC", f"{june['roc_auc']:.3f}")
        cols[2].metric("June top-10% lift", f"{june['top10_lift']:.2f}×")
        st.caption("Top-10% lift uses cohort ranking; it is separate from the single-order cutoff.")
        st.warning("The demo cutoff is 0.5. Historical June recall at this cutoff was 0%; "
                   "it is not an operational alerting policy. Do not interpret a below-cutoff result as safe.")
        frame = pd.DataFrame(stats["evaluation"]).T
        st.bar_chart(frame.loc[["Validation", "June", "July", "August"], ["avg_precision"]])
        st.dataframe(frame[["avg_precision", "roc_auc", "precision_top10", "recall_top10", "brier"]])
        st.write(f"Refit history: {stats['train_rows']:,} orders. "
                 f"Training cutoff: {stats['training_run_date']}.")
        st.caption(stats["evaluation_notes"])
        with st.expander("Model parameters and input schema"):
            st.json({"parameters": stats["best_params"], "features": stats["features"]})


if __name__ == "__main__":
    main()
