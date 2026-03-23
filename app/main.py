import pandas as pd
import numpy as np
import joblib
import json
import time
from fastapi import FastAPI
from typing import List
from app.schemas import HousingData, PredictionResult
from fastapi.concurrency import run_in_threadpool
from app.custom_transformers import CombinedAttributesAdder

app = FastAPI(title="House Price Prediction API")
model = None

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model/model.joblib")
    print("Load model thanh cong")

@app.get("/")
def read_root():
    return {"message": "/docs"}

def predict_with_std(input_df: pd.DataFrame):
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    X_transformed = preprocessor.transform(input_df)

    all_tree_preds = np.array([tree.predict(X_transformed) for tree in rf.estimators_])

    predictions = all_tree_preds.mean(axis=0)   # giống model.predict()
    stds = all_tree_preds.std(axis=0)   # độ phân tán giữa các cây

    return predictions, stds

@app.post("/predict", response_model=List[PredictionResult])
async def predict(data: List[HousingData]):
    start_time = time.time()
    
    input_df = pd.DataFrame([item.model_dump() for item in data])

    predictions, stds = await run_in_threadpool(predict_with_std, input_df)

    duration = f"{(time.time() - start_time) * 1000:.2f} ms"

    results = [
        PredictionResult(
            prediction  = round(float(pred), 2),
            prediction_std = round(float(std), 2),
            prediction_std_low = round(float(pred - 2 * std), 2),
            prediction_std_high = round(float(pred + 2 * std), 2),
            inference_time = duration
        )
        for pred, std in zip(predictions, stds)
    ]

    return results