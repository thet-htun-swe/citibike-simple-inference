import json
from io import StringIO

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Citi Bike Inference UI", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
      html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
      }
      .hero {
        border-radius: 16px;
        padding: 18px 22px;
        background: linear-gradient(135deg, #e8f4ff 0%, #f6fff2 100%);
        border: 1px solid #d9ecff;
      }
      .kpi {
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #eceff3;
        background: #ffffff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">Citi Bike Inference Dashboard</h2>
      <p style="margin-top:8px;margin-bottom:0;">
        Predict <b>trips_out</b>, <b>trips_in</b>, and derived <b>net_flow</b> from trained artifacts.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API Base URL", value="http://localhost:8000")
    st.caption("Run backend first: `uvicorn inference_app.api.main:app --reload`")


def api_get(path: str):
    return requests.get(f"{api_base}{path}", timeout=30)


def api_post(path: str, payload: dict):
    return requests.post(f"{api_base}{path}", json=payload, timeout=60)


tabs = st.tabs(["Model Info", "Reference Lookup", "Manual Features", "Batch CSV"])

with tabs[0]:
    st.subheader("Model Info")
    if st.button("Refresh Model Info"):
        try:
            r = api_get("/model-info")
            r.raise_for_status()
            info = r.json()
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='kpi'><b>Mode</b><br>{info.get('model_mode')}</div>", unsafe_allow_html=True)
            c2.markdown(
                f"<div class='kpi'><b>Feature Count</b><br>{info.get('feature_count')}</div>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<div class='kpi'><b>Artifact Dir</b><br>{info.get('artifact_dir')}</div>",
                unsafe_allow_html=True,
            )
            st.json(info)
        except Exception as e:
            st.error(f"Failed to load model info: {e}")

with tabs[1]:
    st.subheader("Reference Lookup (Station + Datetime)")
    st.caption("Uses saved feature dataset row from backend and predicts without manual feature entry.")

    stations = []
    station_error = None
    try:
        r = requests.get(f"{api_base}/stations", timeout=120)
        if r.status_code == 200:
            stations = r.json().get("stations", [])
        else:
            station_error = f"/stations returned status {r.status_code}"
    except Exception:
        station_error = "Failed to connect to /stations endpoint."

    if station_error:
        st.error(f"Could not load stations: {station_error}")
    if not stations:
        st.warning("No stations were returned. Check API `/model-info` fields: `feature_data_loaded`, `station_count`.")

    station = st.selectbox("Station", options=stations if stations else [""])
    dt = st.text_input("Datetime Hour (YYYY-MM-DD HH:MM:SS)", value="2025-12-20 08:00:00")

    if st.button("Predict from Reference Row"):
        try:
            r = api_get(f"/predict-from-reference?station={station}&datetime_hour={dt}")
            r.raise_for_status()
            out = r.json()
            pred = out.get("prediction", {})
            cols = st.columns(3)
            cols[0].metric("Pred trips_out", f"{pred.get('pred_trips_out', 0):.3f}")
            cols[1].metric("Pred trips_in", f"{pred.get('pred_trips_in', 0):.3f}")
            cols[2].metric("Pred net_flow", f"{pred.get('pred_net_flow', 0):.3f}")

            with st.expander("Feature Row Used"):
                st.json(out.get("features", {}))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tabs[2]:
    st.subheader("Manual Feature Prediction (Single Row)")
    st.caption("Paste JSON dict containing all required feature columns.")
    txt = st.text_area(
        "Feature JSON",
        value='{"station_id": 1, "hour": 8, "lat": 40.74, "lng": -73.99, "is_weekend": 0, "day_of_week": 2, "hour_sin": 0.866, "hour_cos": 0.5, "day_sin": 0.975, "day_cos": -0.223, "trips_out_lag_1h": 2, "trips_out_lag_2h": 1, "trips_out_lag_3h": 1, "trips_out_lag_24h": 3, "trips_out_rolling_mean_3h": 1.33, "trips_in_lag_1h": 1, "trips_in_lag_2h": 2, "trips_in_lag_3h": 1, "trips_in_lag_24h": 4, "trips_in_rolling_mean_3h": 1.33, "lag_net_flow_1h": -1, "lag_net_flow_24h": 1}',
        height=220,
    )
    if st.button("Run Manual Prediction"):
        try:
            payload = {"features": json.loads(txt)}
            r = api_post("/predict", payload)
            r.raise_for_status()
            pred = r.json().get("prediction", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Pred trips_out", f"{pred.get('pred_trips_out', 0):.3f}")
            c2.metric("Pred trips_in", f"{pred.get('pred_trips_in', 0):.3f}")
            c3.metric("Pred net_flow", f"{pred.get('pred_net_flow', 0):.3f}")
        except Exception as e:
            st.error(f"Manual prediction failed: {e}")

with tabs[3]:
    st.subheader("Batch CSV Prediction")
    st.caption("Upload CSV containing feature columns only (or superset including them).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        raw = up.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw))
        st.write("Uploaded rows:", len(df))
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction"):
            try:
                rows = df.to_dict(orient="records")
                r = api_post("/predict-batch", {"rows": rows})
                r.raise_for_status()
                preds = pd.DataFrame(r.json().get("predictions", []))
                out = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                st.success(f"Predicted {len(out)} rows.")
                st.dataframe(out.head(50), use_container_width=True)

                if {"pred_trips_out", "pred_trips_in", "pred_net_flow"}.issubset(out.columns):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg pred_out", f"{out['pred_trips_out'].mean():.3f}")
                    c2.metric("Avg pred_in", f"{out['pred_trips_in'].mean():.3f}")
                    c3.metric("Avg pred_net", f"{out['pred_net_flow'].mean():.3f}")
                    st.bar_chart(out[["pred_trips_out", "pred_trips_in", "pred_net_flow"]].head(100))

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
