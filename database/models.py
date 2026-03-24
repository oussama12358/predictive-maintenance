from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, ForeignKey, Text, Enum as SAEnum
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()
def utcnow():
    return datetime.now(timezone.utc)

# ── 1. Machines ───────────────────────────────────────────────────────────────
class Machine(Base):
    """
    Master registry of all monitored machines.
    One machine can have many sensor readings, predictions, and failure events.
    """
    __tablename__ = "machine_registry"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    machine_id   = Column(String(50), unique=True, nullable=False, index=True)
    product_type = Column(String(1), nullable=False)          # L / M / H
    location     = Column(String(100), nullable=True)
    install_date = Column(DateTime, nullable=True)
    is_active    = Column(Boolean, default=True)
    notes        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=utcnow)

    # Relationships
    sensor_readings  = relationship("SensorReading",  back_populates="machine", cascade="all, delete-orphan")
    failure_events   = relationship("FailureEvent",   back_populates="machine", cascade="all, delete-orphan")
    maintenance_logs = relationship("MaintenanceLog", back_populates="machine", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Machine {self.machine_id} type={self.product_type}>"


# ── 2. Sensor Readings ────────────────────────────────────────────────────────
class SensorReading(Base):
    """
    Stores every sensor payload sent to POST /predict_failure.
    Raw values exactly as received — before feature engineering.
    """
    __tablename__ = "sensor_readings"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    machine_id           = Column(String(50), ForeignKey("machine_registry.machine_id"), nullable=False, index=True)
    timestamp            = Column(DateTime, default=utcnow, index=True)

    # Raw AI4I sensor fields
    air_temp_c           = Column(Float, nullable=False)
    process_temp_c       = Column(Float, nullable=False)
    rotational_speed_rpm = Column(Float, nullable=False)
    torque_nm            = Column(Float, nullable=False)
    tool_wear_min        = Column(Float, nullable=False)

    # Relationship
    machine    = relationship("Machine", back_populates="sensor_readings")
    prediction = relationship("Prediction", back_populates="sensor_reading",
                              uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SensorReading machine={self.machine_id} t={self.timestamp}>"


# ── 3. Predictions ────────────────────────────────────────────────────────────
class Prediction(Base):
    """
    Stores model output for every inference request.
    Linked 1-to-1 to a SensorReading.
    Enables full audit trail: input → output → actual outcome.
    """
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    sensor_reading_id   = Column(Integer, ForeignKey("sensor_readings.id"), unique=True, nullable=False)
    machine_id          = Column(String(50), nullable=False, index=True)
    timestamp           = Column(DateTime, default=utcnow, index=True)

    failure_probability = Column(Float, nullable=False)
    risk_level          = Column(String(10), nullable=False)   # Low / Medium / High
    threshold_used      = Column(Float, nullable=False)
    model_version       = Column(String(30), nullable=False)
    recommendation      = Column(Text, nullable=False)

    # Was this prediction correct? Filled in later when actual outcome is known.
    actual_failure      = Column(Boolean, nullable=True)       # NULL = not yet known
    was_correct         = Column(Boolean, nullable=True)

    # Relationship
    sensor_reading = relationship("SensorReading", back_populates="prediction")

    def __repr__(self):
        return f"<Prediction machine={self.machine_id} risk={self.risk_level} prob={self.failure_probability:.3f}>"


# ── 4. Failure Events ─────────────────────────────────────────────────────────
class FailureEvent(Base):
    """
    Records actual confirmed machine failures.
    Used to close the feedback loop: compare predictions vs reality.
    Enables model performance monitoring over time.
    """
    __tablename__ = "failure_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    machine_id   = Column(String(50), ForeignKey("machine_registry.machine_id"), nullable=False, index=True)
    occurred_at  = Column(DateTime, nullable=False, index=True)
    reported_at  = Column(DateTime, default=utcnow)

    failure_type = Column(String(10), nullable=True)    # TWF / HDF / PWF / OSF / RNF
    description  = Column(Text, nullable=True)
    downtime_hrs = Column(Float, nullable=True)
    repair_cost  = Column(Float, nullable=True)

    # Was there a High-risk prediction within 24h before this failure?
    was_predicted = Column(Boolean, nullable=True)

    machine = relationship("Machine", back_populates="failure_events")

    def __repr__(self):
        return f"<FailureEvent machine={self.machine_id} type={self.failure_type} at={self.occurred_at}>"


# ── 5. Maintenance Log ────────────────────────────────────────────────────────
class MaintenanceLog(Base):
    """
    Records completed maintenance actions.
    Links maintenance to predictions to evaluate if intervention was triggered by model.
    """
    __tablename__ = "maintenance_log"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    machine_id      = Column(String(50), ForeignKey("machine_registry.machine_id"), nullable=False, index=True)
    performed_at    = Column(DateTime, nullable=False)
    logged_at       = Column(DateTime, default=utcnow)

    action_type     = Column(String(50), nullable=False)   # tool_replacement / inspection / repair / overhaul
    triggered_by    = Column(String(20), nullable=False)   # model_alert / scheduled / manual
    technician      = Column(String(100), nullable=True)
    notes           = Column(Text, nullable=True)
    duration_hrs    = Column(Float, nullable=True)
    cost            = Column(Float, nullable=True)

    machine = relationship("Machine", back_populates="maintenance_logs")

    def __repr__(self):
        return f"<MaintenanceLog machine={self.machine_id} action={self.action_type}>"