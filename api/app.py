# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import os

app = FastAPI(title="Fraud Detection API", version="1.0")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# Load model Production khi startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model("models:/FraudDetector/Production")
        print("✅ Model loaded from MLflow")
    except Exception as e:
        print(f"⚠️ Fallback — {e}")

class Transaction(BaseModel):
    features: list[float]  # 30 features

class PredictionResponse(BaseModel):
    fraud: bool
    probability: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    X = np.array(tx.features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    return {"fraud": bool(prob > 0.5), "probability": float(prob)}