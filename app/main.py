import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import HousingData, PredictionResult

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = 'model/model.joblib'
        
        model = joblib.load(model_path)
        
        print(f"--- SUCCESS: Model loaded from {model_path} ---")
    except Exception as e:
        print(f"--- ERROR: Could not load model. Details: {e} ---")

@app.get("/")
async def root():
    return {
        "message": "API đang hoạt động!",
        "model_status": "Ready" if model is not None else "Not Loaded"
    }