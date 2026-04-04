import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

from pipeline.preprocess import full_preprocess

app = FastAPI(title="Azure Demand Forecasting API", version="1.0.0")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Load globally
try:
    model = joblib.load(MODEL_PATH)
    encoded_feature_names = joblib.load(FEATURES_PATH)
    print("Model and feature names loaded successfully.")
except Exception as e:
    print(f"Warning: Model not found at {MODEL_PATH}. It will be loaded on first request.")
    model = None
    encoded_feature_names = None

class DemandRequest(BaseModel):
    region: str
    service_type: str
    date: str
    capacity_allocated: Optional[float] = None
    cost_usd: Optional[float] = None
    availability: Optional[float] = 99.99
    market_demand_index: Optional[float] = 100.0
    gdp_growth: Optional[float] = 2.0
    customer_growth_rate: Optional[float] = 1.5
    industry_mix_index: Optional[float] = 1.0

class DemandResponse(BaseModel):
    prediction: float
    region: str
    service_type: str
    date: str

@app.on_event("startup")
def load_artifacts():
    global model, encoded_feature_names
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            encoded_feature_names = joblib.load(FEATURES_PATH)
        else:
            print("Model files missing. Please run train_and_save.py")

@app.post("/predict", response_model=DemandResponse)
def predict(request: DemandRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Apply the exact same preprocessing as training
        processed_df = full_preprocess(df, is_training=False, encoder_path=ENCODER_PATH)
        
        # Drop columns not expected by model (like date)
        if "date" in processed_df.columns:
            processed_df = processed_df.drop(columns=["date"])
        # Same for demand_units if it leaked in
        if "demand_units" in processed_df.columns:
            processed_df = processed_df.drop(columns=["demand_units"])
            
        # Ensure exact column match with training data
        for col in encoded_feature_names:
            if col not in processed_df.columns:
                processed_df[col] = 0.0
                
        # Reorder to match training exactly
        X = processed_df[encoded_feature_names]
        
        # Predict
        prediction = model.predict(X)[0]
        
        return DemandResponse(
            prediction=float(prediction),
            region=request.region,
            service_type=request.service_type,
            date=request.date
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
