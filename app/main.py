"""
app/main.py - FastAPI application entry point.

Endpoints:
  GET  /health           - Liveness probe
  POST /predict_failure  - Main inference endpoint
"""

import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.schemas import SensorReading, PredictionResponse, HealthResponse
from app.predictor import predictor
from app.logger import get_logger
from database.connection import init_db, get_db
from database import crud

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts and initialize database on startup."""
    logger.info("Starting Predictive Maintenance API - loading artifacts...")
    try:
        predictor.load_artifacts()
        logger.info("Artifacts loaded successfully", extra={"model_loaded": True})
    except FileNotFoundError as e:
        logger.error("Failed to load artifacts", extra={"error": str(e)})

    logger.info("Initializing database...")
    init_db()
    logger.info("Database ready.")
    
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
    """Liveness probe â€” used by Docker HEALTHCHECK and K8s probes."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version="xgb-v1.0",
    )


@app.post("/predict_failure", response_model=PredictionResponse, tags=["Inference"])
async def predict_failure(payload: SensorReading, db: Session = Depends(get_db)):
    """
    Main inference endpoint.
    Returns failure probability, risk level, and maintenance recommendation.
    Saves every request + prediction to the database automatically.
    """
    if not predictor.is_loaded:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    start_time = time.perf_counter()

    try:
        sensor_dict = payload.model_dump()
        result = predictor.predict(sensor_dict)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        machine_id = result["machine_id"] or f"GEN-{int(time.time())}"
        result["machine_id"] = machine_id  # Update the result with the generated ID

        # ── Persist to database ───────────────────────────────────────────────
        # 1. Ensure machine exists in master table
        crud.get_or_create_machine(
            db, machine_id=machine_id,
            product_type=payload.product_type.value
        )

        # 2. Save raw sensor reading
        reading = crud.save_sensor_reading(db, machine_id=machine_id, data={
            "air_temp_c":           payload.air_temp_c,
            "process_temp_c":       payload.process_temp_c,
            "rotational_speed_rpm": payload.rotational_speed_rpm,
            "torque_nm":            payload.torque_nm,
            "tool_wear_min":        payload.tool_wear_min,
        })

        # 3. Save prediction result
        crud.save_prediction(db, sensor_reading_id=reading.id, result=result)
        # ─────────────────────────────────────────────────────────────────────

        logger.info(
            "Prediction completed",
            extra={
                "machine_id":          machine_id,
                "failure_probability": result["failure_probability"],
                "risk_level":          result["risk_level"],
                "latency_ms":          latency_ms,
            }
        )

        return PredictionResponse(**result)

    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e), "traceback": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ── Database query routes ──────────────────────────────────────────────────────

@app.get("/machines", tags=["Database"])
def list_machines(db: Session = Depends(get_db)):
    """List all registered machines."""
    machines = crud.get_all_machines(db)
    return [{"machine_id": m.machine_id, "product_type": m.product_type,
             "is_active": m.is_active, "created_at": m.created_at} for m in machines]


@app.get("/machines/{machine_id}/summary", tags=["Database"])
def machine_summary(machine_id: str, db: Session = Depends(get_db)):
    """Full summary for one machine: readings, predictions, failures, maintenance."""
    summary = crud.get_machine_summary(db, machine_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found.")

    last_pred = summary["last_prediction"]
    last_maint = summary["last_maintenance"]
    return {
        "machine_id":      machine_id,
        "total_readings":  summary["total_readings"],
        "failure_count":   summary["failure_count"],
        "high_risk_count": summary["high_risk_count"],
        "last_prediction": {
            "timestamp":           last_pred.timestamp if last_pred else None,
            "failure_probability": last_pred.failure_probability if last_pred else None,
            "risk_level":          last_pred.risk_level if last_pred else None,
        },
        "last_maintenance": {
            "performed_at": last_maint.performed_at if last_maint else None,
            "action_type":  last_maint.action_type if last_maint else None,
        },
    }


@app.get("/predictions/recent", tags=["Database"])
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    """Fetch the most recent predictions across all machines."""
    preds = crud.get_recent_predictions(db, limit=limit)
    return [{
        "machine_id":          p.machine_id,
        "timestamp":           p.timestamp,
        "failure_probability": p.failure_probability,
        "risk_level":          p.risk_level,
        "recommendation":      p.recommendation,
        "model_version":       p.model_version,
    } for p in preds]


@app.get("/predictions/stats", tags=["Database"])
def prediction_stats(db: Session = Depends(get_db)):
    """Aggregate stats: total predictions, risk distribution, avg probability."""
    return crud.get_prediction_stats(db)


@app.get("/predictions/high-risk", tags=["Database"])
def high_risk_alerts(hours: int = 24, db: Session = Depends(get_db)):
    """All HIGH risk predictions in the last N hours."""
    preds = crud.get_high_risk_predictions(db, hours=hours)
    return [{
        "machine_id":          p.machine_id,
        "timestamp":           p.timestamp,
        "failure_probability": p.failure_probability,
        "recommendation":      p.recommendation,
    } for p in preds]


@app.get("/predictions/trend", tags=["Database"])
def risk_trend(days: int = 7, db: Session = Depends(get_db)):
    """Daily risk level counts for the last N days — for dashboard charts."""
    return crud.get_risk_trend(db, days=days)


@app.get("/machines/{machine_id}/history", tags=["Database"])
def machine_history(machine_id: str, limit: int = 50, db: Session = Depends(get_db)):
    """Sensor reading history for one machine."""
    readings = crud.get_readings_for_machine(db, machine_id, limit=limit)
    return [{
        "timestamp":            r.timestamp,
        "air_temp_c":           r.air_temp_c,
        "process_temp_c":       r.process_temp_c,
        "rotational_speed_rpm": r.rotational_speed_rpm,
        "torque_nm":            r.torque_nm,
        "tool_wear_min":        r.tool_wear_min,
    } for r in readings]
