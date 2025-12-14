# it loads the trained model and uses it to generate predictions.

import joblib
import json
import numpy as np
import pandas as pd
import os

# Load the trained pipeline and threshold
def load_artifacts():
    pipeline = joblib.load("train/artifacts/diabetes_model.joblib")

    with open("train/artifacts/diabetes_threshold.json") as f:
        threshold = json.load(f)["threshold"]

    return pipeline, threshold

# Load at startup
model, threshold = load_artifacts()

# Prediction function
def predict_diabetes(data):
    # Convert input into a DataFrame with the same column names as training
    input_df = pd.DataFrame([{
        "Pregnancies": data.Pregnancies,
        "Glucose": data.Glucose,
        "BloodPressure": data.BloodPressure,
        "SkinThickness": data.SkinThickness,
        "Insulin": data.Insulin,
        "BMI": data.BMI,
        "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
        "Age": data.Age
    }])

    # Get probability directly from the pipeline
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= threshold)  # use your tuned threshold

    # Convert to float for JSON serialisation
    return {
        "probability": float(proba),
        "prediction": pred }



