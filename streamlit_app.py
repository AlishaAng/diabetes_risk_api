import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict" #sends a request to the API endpoint

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
            r = requests.post(API_URL, json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            result = r.json()
            prob = float(result["probability"])
            pred = int(result["prediction"])

            st.subheader("Result")

            # Colour-coded headline
            if pred == 1:
                st.error(f"Higher risk , {prob:.1%} predicted probability")
            else:
                st.success(f"Lower risk , {prob:.1%} predicted probability")

            # Progress bar (nice visual cue)
            st.progress(min(max(prob, 0.0), 1.0))

            # Add a simple risk label (more granular than 0/1)
            if prob < 0.2:
                st.info("Risk level: Very low")
            elif prob < 0.4:
                st.info("Risk level: Low–moderate")
            elif prob < 0.6:
                st.warning("Risk level: Moderate–high")
            else:
                st.warning("Risk level: High")

            with st.expander("See request/response"):
                st.json({"request": payload, "response": result})

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. Is Uvicorn running?\n\n{e}")
