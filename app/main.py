# Sets up web server and API endpoints for diabetes prediction

from fastapi import FastAPI
from app.schema import PatientData
from app.service import predict_diabetes

app = FastAPI(title="Pima Diabetes API") #sets up the web server

# Health check endpoint
@app.get("/")
def health():
    return {"status": "ok"} #useful for confirming the API is running 

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    """
    Accepts a PatientData JSON payload and returns
    probability + binary prediction.
    """
    result = predict_diabetes(data)
    return result

