# Sets up web server and API endpoints for diabetes prediction

from fastapi import FastAPI
from app.schema import PatientData
from app.service import predict_diabetes

app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="Predicts the probability of diabetes using a trained ML model",
    version="1.0.0"
)

# Health check endpoint
@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok"}  #useful for confirming the API is running 

# Prediction endpoint
@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict diabetes risk",
    description="Returns the probability of diabetes and a binary prediction"
)
def predict(data: PatientData):
    return predict_diabetes(data)
