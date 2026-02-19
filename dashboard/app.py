import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data.carobd_adapter import adapt_carobd


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Predictive Vehicle Maintenance", layout="wide")

st.title("🚗 Predictive Vehicle Maintenance System")
st.caption("Upload vehicle telemetry to predict Remaining Useful Life (RUL).")

uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])

if uploaded_file is not None:

    raw_df = pd.read_csv(uploaded_file)
    # 🔥 REMOVE DUPLICATE COLUMNS
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
    raw_df.columns = raw_df.columns.str.strip()


    # ---- Detect carOBD by ENGINE_RPM column ----
    if any(col.upper().startswith("ENGINE_RPM") for col in raw_df.columns):
        st.info("Detected carOBD telemetry format — applying adapter")
        df = adapt_carobd(raw_df)
    else:
        df = raw_df



    st.subheader("📊 Raw Telemetry Preview")
    st.dataframe(df.head())

    if st.button("Run RUL Prediction"):

        with st.spinner("Running predictive maintenance model..."):

            # 🔥 Clean telemetry before sending to API
            df_clean = df.copy()

            df_clean = df_clean.replace([float("inf"), float("-inf")], 0)
            df_clean = df_clean.fillna(0)

            payload = df_clean.to_dict(orient="records")

            response = requests.post(API_URL, json=payload)

            if response.status_code != 200:
                st.error("Prediction API failed")
                st.text(response.text)
                st.stop()

            result = pd.DataFrame(response.json())

        st.success("Prediction successful!")

        # ---------- Smooth curve ----------
        result["RUL_smooth"] = result["predicted_RUL"].rolling(5, min_periods=1).mean()

        latest_rul = result["predicted_RUL"].iloc[-1]
        health_pct = (latest_rul / 125) * 100

        # ---------- Health classification ----------
        if latest_rul > 50:
            status = "🟢 Healthy"
            color = "green"
        elif latest_rul > 25:
            status = "🟡 Warning"
            color = "orange"
        else:
            status = "🔴 Critical"
            color = "red"

        # ---------- Plot ----------
        st.subheader("📉 Predicted RUL Over Time")

        fig, ax = plt.subplots()

        ax.plot(result["cycle"], result["predicted_RUL"], alpha=0.4, label="Raw RUL")
        ax.plot(result["cycle"], result["RUL_smooth"], linewidth=2, label="Smoothed RUL")

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Remaining Useful Life")
        ax.legend()

        st.pyplot(fig)

        # ---------- Metrics ----------
        st.subheader("🧠 Latest Engine Health")

        col1, col2, col3 = st.columns(3)

        col1.metric("RUL (cycles)", f"{latest_rul:.2f}")
        col2.metric("Health %", f"{health_pct:.1f}%")
        col3.metric("Status", status)

        # ---------- Maintenance alert ----------
        if latest_rul < 25:
            st.error("🚨 Maintenance Recommended: Engine nearing failure!")
        elif latest_rul < 30:
            st.error("⚠ Schedule maintenance")
        elif latest_rul < 50:
            st.warning("⚠️ Degradation detected. Schedule inspection.")
        else:
            st.success("✅ Engine operating normally.")
            
            
        st.download_button(
            label="Download Predictions as CSV",
            data=result.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )