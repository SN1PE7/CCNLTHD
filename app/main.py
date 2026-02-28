import pandas as pd
import joblib
import time
from fastapi import FastAPI
from typing import List
from app.schemas import HousingData, PredictionResult

app = FastAPI(title="California House Price Prediction API")

# Bien toan cuc de giu model
model = None

# Load model
@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model/model.joblib")
    print("load thanh cong")

@app.get("/")
def read_root():
    return {"message": "qua /docs"}

# endpoint
@app.post("/predict", response_model=List[PredictionResult])
async def predict(data: List[HousingData]):
    start_time = time.time()
    
    input_df = pd.DataFrame([item.dict() for item in data])

    predictions = model.predict(input_df)

    end_time = time.time()
    duration = f"{(end_time - start_time) * 1000:.2f} ms"

    results = [
        PredictionResult(prediction=pred, inference_time=duration) 
        for pred in predictions
    ]
    
    return results