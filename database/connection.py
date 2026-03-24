"""Environment-based configuration:
  - Development (default) : SQLite at data/predictive_maintenance.db
  - Production (Docker)   : PostgreSQL via DATABASE_URL env variable
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from dotenv import load_dotenv
from database.models import Base

# Load .env file automatically
load_dotenv(dotenv_path=Path(".env"))

# ── Database URL ──────────────────────────────────────────────────────────────
# Set DATABASE_URL in .env for your environment:
#   Dev  : sqlite:///data/predictive_maintenance.db
#   Prod : postgresql://user:password@localhost:5432/pdm_db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///data/predictive_maintenance.db"
)

# ── Engine configuration ──────────────────────────────────────────────────────
def create_db_engine():
    if DATABASE_URL.startswith("sqlite"):
        # SQLite requires special config for concurrent use
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,    # Set True to log all SQL statements
        )
        # Enable WAL mode for better SQLite concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # PostgreSQL / other databases
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   
            echo=False,
        )
    return engine

engine = create_db_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def init_db() -> None:
    """
    Create all tables if they don't exist.
    Called once at application startup.
    Safe to call multiple times — does nothing if tables already exist.
    """
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.
    Automatically closes the session after the request completes.

    Usage:
        @app.post("/predict_failure")
        def predict(payload: SensorReading, db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()