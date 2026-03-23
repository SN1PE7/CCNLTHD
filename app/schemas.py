from pydantic import BaseModel
from typing import List

class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }

class PredictionResult(BaseModel):
    prediction:          float  # kq
    prediction_std:      float  # độ lệch chuẩn
    prediction_std_low:  float  # thấp nhất dự tính
    prediction_std_high: float  # cao nhất dự tính
    inference_time:      str    # thời gian xử lý request