"""Streamlit monitoring dashboard.

Four views:

1. Forecast: historical actuals plus model predictions and prediction intervals.
2. Explainability: SHAP summary and force plots.
3. Drift: Evidently report plus monthly MAPE trend.
4. Model Registry: MLflow runs table with metrics.

The dashboard consumes the FastAPI service for fresh predictions, never the
model directly. The separation enforces a single inference path through the
API rather than two parallel ones.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.set_page_config(
    page_title="MLOps Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def fetch_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/model/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_forecast(start_ts: str, horizon: int) -> list[dict] | None:
    try:
        payload = {"start_timestamp": start_ts, "horizon_hours": horizon}
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["forecast"]
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


@st.cache_data(ttl=600)
def load_predictions() -> pd.DataFrame | None:
    path = "data/07_model_output/predictions.parquet"
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df
    except FileNotFoundError:
        return None


@st.cache_data(ttl=600)
def load_shap_values() -> pd.DataFrame | None:
    try:
        return pd.read_parquet("data/06_models/shap_values.parquet")
    except FileNotFoundError:
        return None


@st.cache_data(ttl=600)
def load_drift_results() -> pd.DataFrame | None:
    try:
        return pd.read_csv("data/08_reporting/drift/concept_drift_results.csv")
    except FileNotFoundError:
        return None


@st.cache_data(ttl=120)
def load_mlflow_runs() -> pd.DataFrame | None:
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(MLFLOW_URI)
        runs = mlflow.search_runs(
            experiment_names=["ElectricityForecast"],
            order_by=["metrics.val_mape ASC"],
            max_results=20,
        )
        return runs if not runs.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ MLOps Forecast")
page = st.sidebar.radio("View", ["Forecast", "Explainability", "Drift", "Model Registry"])

model_info = fetch_model_info()
if model_info:
    st.sidebar.success(
        f"Model: **{model_info['model_name']}** v{model_info['model_version']}\n\n"
        f"Stage: `{model_info['model_stage']}`"
    )
else:
    st.sidebar.warning("API not reachable. Some views may be limited")


# ---------------------------------------------------------------------------
# Page: Forecast
# ---------------------------------------------------------------------------
if page == "Forecast":
    st.title("⚡ Electricity Load Forecast")
    st.caption("Germany hourly load (MW). OPSD dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        horizon = st.slider("Forecast horizon (hours)", 1, 168, 24, step=1)
    with col2:
        start_date = st.date_input("Forecast start date", value=datetime(2020, 6, 1).date())

    start_ts = (
        datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
    )

    with st.spinner("Fetching forecast from API..."):
        forecast_data = fetch_forecast(start_ts, horizon)

    # Historical actuals from pipeline output
    historical = load_predictions()

    fig = go.Figure()

    if historical is not None and not historical.empty:
        plot_hist = historical.tail(24 * 14)  # last 2 weeks
        fig.add_trace(
            go.Scatter(
                x=plot_hist.index,
                y=plot_hist["load_mw_actual"],
                name="Actual",
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_hist.index,
                y=plot_hist["load_mw_predicted"],
                name="Model prediction",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            )
        )

    if forecast_data:
        fc_df = pd.DataFrame(forecast_data)
        fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])
        fig.add_trace(
            go.Scatter(
                x=fc_df["timestamp"],
                y=fc_df["load_mw_predicted"],
                name="API forecast",
                line=dict(color="#2ca02c", width=2),
            )
        )

    fig.update_layout(
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if historical is not None:
        test_mape = (historical["abs_error"] / historical["load_mw_actual"].abs() * 100).mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Test MAPE", f"{test_mape:.2f}%")
        col2.metric("Test rows", len(historical))
        if model_info:
            col3.metric("Model version", model_info["model_version"])


# ---------------------------------------------------------------------------
# Page: Explainability
# ---------------------------------------------------------------------------
elif page == "Explainability":
    st.title("🔍 Model Explainability. SHAP")
    st.caption(
        "SHAP values computed on LightGBM model (TreeExplainer, 500-row sample from validation set)"
    )

    shap_df = load_shap_values()

    if shap_df is None:
        st.info("Run the model_train pipeline first to generate SHAP values.")
    else:
        # SHAP summary: mean absolute SHAP per feature
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True).tail(20)
        fig = px.bar(
            x=mean_abs_shap.values,
            y=mean_abs_shap.index,
            orientation="h",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            title="Global Feature Importance (mean |SHAP|)",
            color=mean_abs_shap.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Per-prediction force plot
        st.subheader("Per-prediction SHAP values")
        idx = st.slider("Select row index", 0, len(shap_df) - 1, 0)
        row = shap_df.iloc[idx].sort_values(key=abs, ascending=False).head(15)
        fig2 = px.bar(
            x=row.values,
            y=row.index,
            orientation="h",
            labels={"x": "SHAP contribution", "y": "Feature"},
            title=f"SHAP force plot. Row {idx}",
            color=row.values,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Drift
# ---------------------------------------------------------------------------
elif page == "Drift":
    st.title("📊 Drift Monitoring")
    st.caption(
        "Feature drift (Evidently) on 2019 reference vs 2020 COVID lockdown period.\n"
        "Both the no-drift (2019) and drift-detected (2020 Q1–Q2) cases are shown."
    )

    concept_drift = load_drift_results()

    if concept_drift is not None:
        st.subheader("Concept Drift: Monthly MAPE over Time")
        fig = px.line(
            concept_drift,
            x="month",
            y="mape",
            markers=True,
            title="Monthly MAPE. Test period (2020)",
            labels={"mape": "MAPE (%)", "month": "Month"},
        )
        if "drift_flag" in concept_drift.columns:
            drifted = concept_drift[concept_drift["drift_flag"] == True]  # noqa: E712
            fig.add_scatter(
                x=drifted["month"],
                y=drifted["mape"],
                mode="markers",
                marker=dict(color="red", size=12, symbol="x"),
                name="Drift detected",
            )
        fig.add_hline(y=concept_drift["mape"].mean(), line_dash="dot", annotation_text="Mean MAPE")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the data_drifts pipeline to generate drift results.")

    st.subheader("Feature Drift Report (Evidently)")
    html_path = "data/08_reporting/drift/evidently_drift_report.html"
    try:
        with open(html_path, encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    except FileNotFoundError:
        st.info("Run the data_drifts pipeline to generate the Evidently HTML report.")

    with st.expander("Drift experiment design"):
        st.markdown(
            """
**Reference period:** 2019 (stable pre-COVID baseline)
**Analysis period:** 2020 H1 (COVID-19 lockdown. German mobility reduced ~40%)

The COVID-19 lockdown period is used deliberately as a known structural break.
Industrial and commercial electricity demand dropped significantly while residential
demand increased. A genuine concept shift that a well-designed monitor should detect.

Expected result:
- Reference vs reference: **no drift** (p > 0.05 on all features)
- Reference vs COVID window: **drift detected** (significant KS statistic on load, time-of-day patterns)
"""
        )


# ---------------------------------------------------------------------------
# Page: Model Registry
# ---------------------------------------------------------------------------
elif page == "Model Registry":
    st.title("🗂️ MLflow Model Registry")
    st.caption(f"Tracking server: {MLFLOW_URI}")

    runs_df = load_mlflow_runs()

    if runs_df is None:
        st.warning(
            "Could not connect to MLflow. "
            "Start the tracking server with `make mlflow-ui` or `make serve`."
        )
    else:
        display_cols = [
            c
            for c in runs_df.columns
            if any(
                kw in c
                for kw in [
                    "run_id",
                    "status",
                    "start",
                    "params.model_type",
                    "metrics.val_mape",
                    "metrics.val_rmse",
                ]
            )
        ]
        st.dataframe(
            runs_df[display_cols].rename(
                columns=lambda c: c.replace("metrics.", "").replace("params.", "")
            ),
            use_container_width=True,
            height=300,
        )

    if model_info:
        st.subheader("Current Production Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("Version", model_info["model_version"])
        col2.metric(
            "val_mape",
            f"{model_info['metrics'].get('val_mape', 'N/A'):.2f}%"
            if "val_mape" in model_info.get("metrics", {})
            else "N/A",
        )
        col3.metric(
            "val_rmse",
            f"{model_info['metrics'].get('val_rmse', 'N/A'):.0f} MW"
            if "val_rmse" in model_info.get("metrics", {})
            else "N/A",
        )

        st.info(
            "**Promotion workflow:** Staging → Production requires manual review.\n"
            "In a production system, the 'Promote' button below would trigger "
            "a model version transition in the MLflow registry. A metadata change, "
            "not a redeploy."
        )
        st.button(
            "⬆️ Promote Staging → Production",
            disabled=True,
            help="Manual gate: review metrics before promoting",
        )
