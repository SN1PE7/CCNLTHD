from typing import Any, List, Optional

from pydantic import BaseModel, Field, conlist, validator


ALLOWED_OCEAN_PROXIMITY = {
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
}

class HousingData(BaseModel):
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)
    housing_median_age: float = Field(..., ge=0)
    total_rooms: float = Field(..., ge=0)
    total_bedrooms: float = Field(..., ge=0)
    population: float = Field(..., ge=0)
    households: float = Field(..., ge=0)
    median_income: float = Field(..., ge=0)
    ocean_proximity: str

    @validator("ocean_proximity")
    def validate_ocean_proximity(cls, value: str) -> str:
        normalized_value = value.strip().upper()
        if normalized_value not in ALLOWED_OCEAN_PROXIMITY:
            allowed_values = ", ".join(sorted(ALLOWED_OCEAN_PROXIMITY))
            raise ValueError(f"ocean_proximity must be one of: {allowed_values}")
        return normalized_value

    class Config:
        schema_extra = {
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


class BatchPredictionRequest(BaseModel):
    items: conlist(HousingData, min_length=1, max_length=128)

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "longitude": -122.23,
                        "latitude": 37.88,
                        "housing_median_age": 41.0,
                        "total_rooms": 880.0,
                        "total_bedrooms": 129.0,
                        "population": 322.0,
                        "households": 126.0,
                        "median_income": 8.3252,
                        "ocean_proximity": "NEAR BAY",
                    }
                ]
            }
        }


class PredictionItem(BaseModel):
    prediction: float


class PredictionMeta(BaseModel):
    batch_size: int
    elapsed_ms: float


class BatchPredictionResponse(BaseModel):
    meta: PredictionMeta
    predictions: List[PredictionItem]


class ErrorDetail(BaseModel):
    loc: List[Any]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    request_id: str