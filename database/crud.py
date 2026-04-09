from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from database.models import Machine, SensorReading, Prediction, FailureEvent, MaintenanceLog

# ═══════════════════════════════════════════════════════════════════════════════
# MACHINES
# ═══════════════════════════════════════════════════════════════════════════════

def get_or_create_machine(db: Session, machine_id: str, product_type: str) -> Machine:
    """
    Fetch existing machine or create a new one.
    Called automatically on every prediction for unknown machine IDs.
    """
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        machine = Machine(machine_id=machine_id, product_type=product_type)
        db.add(machine)
        db.commit()
        db.refresh(machine)
    return machine


def get_all_machines(db: Session, active_only: bool = True) -> list:
    query = db.query(Machine)
    if active_only:
        query = query.filter(Machine.is_active == True)
    return query.order_by(Machine.machine_id).all()


def get_machine_summary(db: Session, machine_id: str) -> dict:
    """
    Returns a full summary for one machine:
    total readings, last prediction, failure count, last maintenance.
    """
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        return {}

    total_readings = db.query(func.count(SensorReading.id))\
        .filter(SensorReading.machine_id == machine_id).scalar()

    last_prediction = db.query(Prediction)\
        .filter(Prediction.machine_id == machine_id)\
        .order_by(desc(Prediction.timestamp)).first()

    failure_count = db.query(func.count(FailureEvent.id))\
        .filter(FailureEvent.machine_id == machine_id).scalar()

    last_maintenance = db.query(MaintenanceLog)\
        .filter(MaintenanceLog.machine_id == machine_id)\
        .order_by(desc(MaintenanceLog.performed_at)).first()

    high_risk_count = db.query(func.count(Prediction.id))\
        .filter(Prediction.machine_id == machine_id,
                Prediction.risk_level == "High").scalar()

    return {
        "machine":          machine,
        "total_readings":   total_readings,
        "last_prediction":  last_prediction,
        "failure_count":    failure_count,
        "high_risk_count":  high_risk_count,
        "last_maintenance": last_maintenance,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR READINGS
# ═══════════════════════════════════════════════════════════════════════════════

def save_sensor_reading(db: Session, machine_id: str, data: dict) -> SensorReading:
    """
    Persist a sensor reading payload to the database.
    Called immediately when /predict_failure receives a valid request.
    """
    reading = SensorReading(
        machine_id           = machine_id,
        air_temp_c           = data["air_temp_c"],
        process_temp_c       = data["process_temp_c"],
        rotational_speed_rpm = data["rotational_speed_rpm"],
        torque_nm            = data["torque_nm"],
        tool_wear_min        = data["tool_wear_min"],
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_readings_for_machine(db: Session, machine_id: str,
                              limit: int = 100, offset: int = 0) -> list:
    return db.query(SensorReading)\
        .filter(SensorReading.machine_id == machine_id)\
        .order_by(desc(SensorReading.timestamp))\
        .offset(offset).limit(limit).all()

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_prediction(db: Session, sensor_reading_id: int, result: dict) -> Prediction:
    """
    Persist the model's prediction result.
    Always called immediately after save_sensor_reading().
    """
    pred = Prediction(
        sensor_reading_id   = sensor_reading_id,
        machine_id          = result["machine_id"],
        failure_probability = result["failure_probability"],
        risk_level          = result["risk_level"],
        threshold_used      = result["threshold_used"],
        model_version       = result["model_version"],
        recommendation      = result["recommendation"],
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_recent_predictions(db: Session, limit: int = 50) -> list:
    """Fetch the most recent predictions across all machines."""
    return db.query(Prediction)\
        .order_by(desc(Prediction.timestamp))\
        .limit(limit).all()


def get_high_risk_predictions(db: Session, hours: int = 24) -> list:
    """Fetch all HIGH risk predictions in the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return db.query(Prediction)\
        .filter(Prediction.risk_level == "High",
                Prediction.timestamp >= since)\
        .order_by(desc(Prediction.timestamp)).all()


def get_prediction_stats(db: Session) -> dict:
    """Aggregate stats for the dashboard overview panel."""
    total = db.query(func.count(Prediction.id)).scalar()
    high  = db.query(func.count(Prediction.id))\
              .filter(Prediction.risk_level == "High").scalar()
    med   = db.query(func.count(Prediction.id))\
              .filter(Prediction.risk_level == "Medium").scalar()
    low   = db.query(func.count(Prediction.id))\
              .filter(Prediction.risk_level == "Low").scalar()
    avg_prob = db.query(func.avg(Prediction.failure_probability)).scalar() or 0.0

    return {
        "total":        total,
        "high_risk":    high,
        "medium_risk":  med,
        "low_risk":     low,
        "avg_probability": round(float(avg_prob), 4),
    }


def get_risk_trend(db: Session, days: int = 7) -> list:
    """
    Returns hourly risk level counts for the last N days.
    Used for the trend chart in the dashboard.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    results = db.query(
        func.strftime('%Y-%m-%d %H', Prediction.timestamp).label("date"),
        Prediction.risk_level,
        func.count(Prediction.id).label("count")
    ).filter(Prediction.timestamp >= since)\
     .group_by(func.strftime('%Y-%m-%d %H', Prediction.timestamp), Prediction.risk_level)\
     .order_by(func.strftime('%Y-%m-%d %H', Prediction.timestamp)).all()

    return [{"date": str(r.date), "risk_level": r.risk_level, "count": r.count}
            for r in results]

# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

def log_failure_event(db: Session, machine_id: str, failure_type: str,
                      occurred_at: datetime, description: str = None,
                      downtime_hrs: float = None, repair_cost: float = None) -> FailureEvent:
    """Register a confirmed machine failure event."""
    event = FailureEvent(
        machine_id   = machine_id,
        occurred_at  = occurred_at,
        failure_type = failure_type,
        description  = description,
        downtime_hrs = downtime_hrs,
        repair_cost  = repair_cost,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_failure_events(db: Session, machine_id: Optional[str] = None,
                       limit: int = 50) -> list:
    query = db.query(FailureEvent)
    if machine_id:
        query = query.filter(FailureEvent.machine_id == machine_id)
    return query.order_by(desc(FailureEvent.occurred_at)).limit(limit).all()

# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE LOG
# ═══════════════════════════════════════════════════════════════════════════════

def log_maintenance(db: Session, machine_id: str, action_type: str,
                    performed_at: datetime, triggered_by: str = "manual",
                    technician: str = None, notes: str = None,
                    duration_hrs: float = None, cost: float = None) -> MaintenanceLog:
    """Register a completed maintenance action."""
    log = MaintenanceLog(
        machine_id   = machine_id,
        performed_at = performed_at,
        action_type  = action_type,
        triggered_by = triggered_by,
        technician   = technician,
        notes        = notes,
        duration_hrs = duration_hrs,
        cost         = cost,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_maintenance_history(db: Session, machine_id: str, limit: int = 20) -> list:
    return db.query(MaintenanceLog)\
        .filter(MaintenanceLog.machine_id == machine_id)\
        .order_by(desc(MaintenanceLog.performed_at))\
        .limit(limit).all()