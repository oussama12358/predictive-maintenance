"""
app/main.py - FastAPI application entry point.

Endpoints:
  GET  /health           - Liveness probe
  POST /predict_failure  - Main inference endpoint
"""

import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import SensorReading, PredictionResponse, HealthResponse
from app.predictor import predictor
from app.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup. First request will not be slow."""
    logger.info("Starting Predictive Maintenance API - loading artifacts...")
    try:
        predictor.load_artifacts()
        logger.info("Artifacts loaded successfully", extra={"model_loaded": True})
    except FileNotFoundError as e:
        logger.error("Failed to load artifacts", extra={"error": str(e)})
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "ML inference service for industrial equipment failure prediction. "
        "Predicts failure probability within 7 days from sensor readings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Operations"])
def home():
    """Home endpoint providing API information."""
    return {"message": "Predictive Maintenance API is running"}


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Liveness probe — used by Docker HEALTHCHECK and K8s probes."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version="xgb-v1.0",
    )


@app.post("/predict_failure", response_model=PredictionResponse, tags=["Inference"])
async def predict_failure(payload: SensorReading):
    """
    Main inference endpoint.
    Returns failure probability, risk level, and maintenance recommendation.
    """
    if not predictor.is_loaded:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    start_time = time.perf_counter()

    try:
        sensor_dict = payload.model_dump()
        result = predictor.predict(sensor_dict)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Prediction completed",
            extra={
                "machine_id": result["machine_id"],
                "failure_probability": result["failure_probability"],
                "risk_level": result["risk_level"],
                "latency_ms": latency_ms,
            }
        )

        return PredictionResponse(**result)

    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e), "traceback": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")