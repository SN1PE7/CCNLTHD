from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionItem,
    PredictionMeta,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("model/model.joblib")
    yield

app = FastAPI(
    title="House Price Prediction API",
    description="Asynchronous batch inference API for California housing model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Invalid request payload",
        details=exc.errors(),
        request_id=request.state.request_id,
    )
    return JSONResponse(status_code=422, content=payload.dict())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    payload = ErrorResponse(
        error_code="HTTP_ERROR",
        message=error_message,
        request_id=request.state.request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _: Exception):
    payload = ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message="Unexpected server error",
        request_id=request.state.request_id,
    )
    return JSONResponse(status_code=500, content=payload.dict())

@app.get("/")
def read_root():
    return {"message": "Open /docs for Swagger UI"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(payload: BatchPredictionRequest):
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start = perf_counter()
    input_df = pd.DataFrame([item.dict() for item in payload.items])
    raw_predictions = await run_in_threadpool(app.state.model.predict, input_df)
    elapsed_ms = round((perf_counter() - start) * 1000, 3)

    predictions = [PredictionItem(prediction=float(value)) for value in raw_predictions]

    return BatchPredictionResponse(
        meta=PredictionMeta(batch_size=len(predictions), elapsed_ms=elapsed_ms),
        predictions=predictions,
    )