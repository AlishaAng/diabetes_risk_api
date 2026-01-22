import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL.rstrip('/')}/predict"

def explain_inputs(payload: dict) -> list[str]:
    reasons = []

    if payload["Glucose"] >= 140:
        reasons.append("Glucose is high (≥140), which strongly increases risk.")
    elif payload["Glucose"] >= 110:
        reasons.append("Glucose is elevated (110–139), which increases risk.")

    if payload["BMI"] >= 30:
        reasons.append("BMI is in the obese range (≥30), which increases risk.")
    elif payload["BMI"] >= 25:
        reasons.append("BMI is above the healthy range (≥25), which can increase risk.")

    if payload["Age"] >= 35:
        reasons.append("Age is 35+, which increases baseline risk.")

    if payload["DiabetesPedigreeFunction"] >= 0.8:
        reasons.append("Family-history proxy (DPF) is relatively high, increasing risk.")

    if payload["BloodPressure"] >= 90:
        reasons.append("Blood pressure is high (≥90 diastolic), which can correlate with metabolic risk.")

    if not reasons:
        reasons.append("No obvious high-risk signals based on typical cut-offs — risk may be driven by combinations of factors.")

    return reasons


st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Checker")
st.write("Enter patient measurements to estimate diabetes risk.")


with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.5, step=0.1, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f")
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)

    submitted = st.form_submit_button("Predict risk")


if submitted:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    try:
        with st.spinner("Calculating…"):
            resp = requests.post(PREDICT_URL, json=payload, timeout=10)

        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            result = resp.json()
            prob = float(result["probability"])
            pred = int(result["prediction"])

            st.subheader("Result")
            st.metric("Predicted probability", f"{prob:.1%}")

            # Clear colour-coded interpretation
            if prob < 0.2:
                st.success("Risk level: Very low")
            elif prob < 0.4:
                st.info("Risk level: Low–moderate")
            elif prob < 0.6:
                st.warning("Risk level: Moderate–high")
            else:
                st.error("Risk level: High")

        with st.expander("Why this result? (risk explanation)"):
            reasons = explain_inputs(payload)
            for reason in reasons[:5]:
                st.write("• " + reason)
            st.caption("Note: this explanation is based on common clinical cut-offs, not a guaranteed model attribution")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. Is the API container running?\n\n{e}")
