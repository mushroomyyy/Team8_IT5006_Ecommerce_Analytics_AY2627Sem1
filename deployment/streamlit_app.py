"""Operations-facing Streamlit app for the regression deployment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
from src.data import load_olist_data  # noqa: E402
from src.features import add_extra_features, build_feature_table  # noqa: E402
try:
    from .model_service import ModelService, load_stats  # noqa: E402
except ImportError:  # direct `streamlit deployment/streamlit_app.py`
    from model_service import ModelService, load_stats  # type: ignore # noqa: E402


st.set_page_config(page_title="Delivery Operations", page_icon="📦", layout="wide")
st.markdown("""
<style>
.app-card { padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,.25);
            border-radius: .7rem; margin-bottom: 1rem; }
.small-note { color: #8b949e; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📦 Delivery Operations")
st.sidebar.caption("E-commerce analytics deployment")
st.sidebar.markdown("**Model**  ")
st.sidebar.write("R1 · OLS (all 24 features)")
st.sidebar.markdown("**Target**  ")
st.sidebar.write("Days before or after promised delivery")
st.sidebar.markdown("**Architecture**  ")
st.sidebar.write("Streamlit UI · FastAPI-ready backend")
st.sidebar.divider()
st.sidebar.info("Use a historical order to simulate a new operational case, or enter approval-time details manually.")

st.title("Delivery timing prediction")
st.caption("Identify orders that may need follow-up before the promised delivery date.")


@st.cache_resource
def get_service() -> ModelService:
    return ModelService()


@st.cache_data
def get_samples() -> pd.DataFrame:
    olist = load_olist_data(verbose=False)
    return add_extra_features(build_feature_table(olist), olist)


try:
    service = get_service()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

stats = load_stats()
features = service.feature_cols
numeric = stats["features"]["numeric"]
categorical = stats["features"]["categorical"]

prediction_tab, stats_tab = st.tabs(["🔎 Prediction", "📊 Model Stats"])

with prediction_tab:
    st.subheader("Order follow-up check")
    st.markdown('<div class="small-note">The model only uses information available at order approval.</div>', unsafe_allow_html=True)
    samples = get_samples()
    sample_ids = samples["order_id"].dropna().astype(str).tolist()
    selected_id = st.selectbox("Choose a historical order or enter details", ["Manual entry"] + sample_ids[:500])
    if selected_id == "Manual entry":
        defaults = samples[features].median(numeric_only=True).to_dict()
        defaults.update({column: samples[column].dropna().astype(str).mode().iloc[0]
                         for column in categorical})
    else:
        defaults = samples.loc[samples["order_id"].astype(str).eq(selected_id), features].iloc[0].to_dict()

    with st.form("prediction_form"):
        st.markdown('<div class="app-card"><h3>Approval-time order details</h3><p class="small-note">Same 24 features as the Phase 2 regression pipeline.</p></div>', unsafe_allow_html=True)
        values = {}
        with st.expander("Order, timing and logistics", expanded=True):
            cols = st.columns(3)
            for index, column in enumerate(numeric):
                value = defaults.get(column, 0.0)
                values[column] = cols[index % 3].number_input(column, value=float(value) if pd.notna(value) else 0.0,
                                                              key=f"num_{column}")
        with st.expander("Categorical details"):
            cols = st.columns(3)
            for index, column in enumerate(categorical):
                default = str(defaults.get(column, "Unknown"))
                values[column] = cols[index].selectbox(column, [default, "Unknown"], index=0,
                                                       key=f"cat_{column}")
        submitted = st.form_submit_button("🚚 Check delivery timing", type="primary")

    if submitted:
        try:
            result = service.predict(values)
            days = result["predicted_days_from_promise"]
            status = result["delivery_status"]
            result_cols = st.columns(3)
            result_cols[0].metric("Predicted deviation", f"{days:+.1f} days")
            result_cols[1].metric("Operational status", "Follow up" if status == "late" else "Monitor")
            result_cols[2].metric("Model", result["model_code"])
            if status == "late":
                st.error(f"Follow-up recommended: predicted {days:.1f} days late.")
            else:
                st.success(f"No immediate follow-up signal: predicted {abs(days):.1f} days before/on the promise.")
        except ValueError as exc:
            st.error(str(exc))

with stats_tab:
    st.subheader("Model performance & monitoring")
    st.info("These metrics come from held-out Phase 2 evaluation, not from predictions made in this app.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Validation RMSE", f"{stats['evaluation']['validation_rmse']:.2f} days")
    metric_cols[1].metric("June Test RMSE", f"{stats['evaluation']['june_test_rmse']:.2f} days")
    metric_cols[2].metric("June Test MAE", f"{stats['evaluation']['june_test_mae']:.2f} days")
    metric_cols[3].metric("Training rows", f"{stats['train_rows']:,}")
    st.write({"Model": stats["model_code"], "Version": stats["model_version"],
              "Best parameters": stats["best_params"]})

    comparison_path = ROOT / "phase2/results/simple_to_complex/regression/regression_model_comparison_5.csv"
    comparison = pd.read_csv(comparison_path)
    comparison.columns = comparison.columns.str.strip()
    chart = comparison[comparison["Code"].astype(str).str.strip().isin(["R1", "R2", "R3", "R3d"])].copy()
    chart["Code"] = chart["Code"].astype(str).str.strip()
    chart["June RMSE"] = chart["June"]
    st.plotly_chart(px.bar(chart, x="Code", y="June RMSE", title="June (Test) RMSE by model",
                           labels={"June RMSE": "RMSE (days)"}), use_container_width=True)
    st.caption("R1 is deployed because the Phase 2 report selects it as the regression model; the tuned RF is retained as a challenger in the evaluation results.")
