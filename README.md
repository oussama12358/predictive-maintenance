# Predictive Maintenance System

A production-grade end-to-end ML system for industrial equipment failure prediction.
Predicts the probability of machine failure within the next 7 days based on sensor data.

## Key Highlights

- End-to-end ML system (training → deployment → monitoring)
- Production-ready FastAPI service
- SQL-backed prediction tracking & analytics
- Robustness evaluation (bootstrap)
- Dockerized deployment

## System Architecture

```mermaid
flowchart LR
    A[Sensor Data] --> B[Feature Engineering]
    B --> C[XGBoost Model]
    C --> D[FastAPI Inference Service]
    D --> E[Prediction API]
    E --> F[Database Layer]
    F --> G[Streamlit Dashboard]
    G --> H[User Monitoring Interface]
```
Pipeline Overview:

Data Generation
    ↓
Feature Engineering
    ↓
Model Training (XGBoost)
    ↓
FastAPI Inference Service
    ↓
Streamlit Dashboard

Artifacts:
model/artifacts/
├── model.pkl
├── scaler.pkl
└── threshold.json
```

## Project Structure

predictive_maintenance/
├── app/                    # FastAPI inference service
│   ├── main.py             # Routes and lifespan
│   ├── schemas.py          # Pydantic I/O models
│   ├── predictor.py        # Model loading + inference
│   └── logger.py           # JSON structured logging
├── database/               # Database layer
│   ├── models.py           # SQLAlchemy ORM models
│   ├── crud.py             # Database operations
│   ├── connection.py       # Database configuration
│   └── init.py             # Database initialization
├── model/                  # ML pipeline
│   ├── train.py            # Full training pipeline
│   ├── features.py         # Feature engineering (shared)
│   └── evaluate.py         # Metrics + threshold selection
├── data/
│   └── generate_dataset.py # Synthetic dataset generator
├── dashboard/
│   └── app.py              # Streamlit UI
├── tests/
│   ├── test_api.py         # API integration tests
│   └── test_pipeline.py    # Pipeline unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart (Windows Native)

### 1. Prerequisites

Install Python 3.11 from https://python.org/downloads

### 2. Create virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Download the AI4I 2020 Dataset

**Option A — Auto download (recommended):**
```cmd
python data/load_ai4i.py
```

**Option B — Manual download from Kaggle:**
1. Go to: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
2. Download `ai4i2020.csv`
3. Place it in `data/raw/ai4i2020.csv`
4. Run: `python data/load_ai4i.py --source csv`

Expected output:
```
✅ Downloaded: 10,000 rows x 14 columns
   Failure rate: 3.4%  (339 failures)
   TWF:  46 cases
   HDF: 115 cases
   PWF:  95 cases
   OSF:  98 cases
   RNF:  19 cases
```

### 5. Train the model

```cmd
python -m model.train
```

Expected output:
```
[1/7] Loading data...
✅ Data loaded: 10,000 rows x 13 columns
   Failure rate: 3.4%
[4/7] Applying SMOTE to training set...
[6/7] Training XGBoost classifier...
   CV ROC-AUC: 0.9380 +/- 0.0091
   Optimal threshold (F2): 0.76
   ROC-AUC: 0.9412
✅ Pipeline complete. All artifacts saved.
```

### 6. Start the API

```cmd
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Test it:
```cmd
curl -X POST http://localhost:8000/predict_failure ^
  -H "Content-Type: application/json" ^
  -d "{\"temperature_C\":85.3,\"vibration_mms\":4.7,\"pressure_bar\":6.8,\"runtime_hours\":5200,\"rpm\":1430,\"oil_level_pct\":45,\"error_count_24h\":3,\"ambient_temp_C\":22}"
```

Or open Swagger UI: http://localhost:8000/docs

### 7. Start the dashboard

In a second terminal:
```cmd
.venv\Scripts\activate
streamlit run dashboard/app.py
```

Open: http://localhost:8501

### 8. Run tests

```cmd
pytest tests/ -v
```

---

## Docker Deployment

### Install Docker Desktop for Windows
Download from: https://docs.docker.com/desktop/install/windows-install/

Enable WSL2 backend during installation (recommended).

### Build and run

```cmd
docker build -t pdm-api:latest .
docker run -p 8000:8000 pdm-api:latest
```

### Or use docker compose (API + Dashboard together)

```cmd
docker compose up --build
```

---

## API Reference

### POST /predict_failure

**Request body:**
```json
{
  "machine_id":           "MIL-0042",
  "product_type":         "M",
  "air_temp_c":           25.1,
  "process_temp_c":       36.4,
  "rotational_speed_rpm": 1551.0,
  "torque_nm":            42.8,
  "tool_wear_min":        108.0
}
```

**Response:**
```json
{
  "machine_id":          "MIL-0042",
  "failure_probability": 0.0821,
  "risk_level":          "Low",
  "threshold_used":      0.76,
  "recommendation":      "No immediate action required. Continue normal monitoring.",
  "top_risk_factors": [
    "wear_torque_interaction: 4622.4  (importance: 0.287)",
    "tool_wear_min: 108.0  (importance: 0.231)",
    "speed_torque_index: 66381.8  (importance: 0.187)"
  ],
  "model_version": "xgb-v1.0"
}
```

### GET /health

Returns API liveness + model load status.

---

## Database Layer

The system uses SQLAlchemy ORM with dual database support:

### Database Schema

```sql
-- Machines master table
CREATE TABLE machines (
    id INTEGER PRIMARY KEY,
    machine_id VARCHAR(50) UNIQUE NOT NULL,
    product_type VARCHAR(10) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sensor readings
CREATE TABLE sensor_readings (
    id INTEGER PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    air_temp_c REAL NOT NULL,
    process_temp_c REAL NOT NULL,
    rotational_speed_rpm REAL NOT NULL,
    torque_nm REAL NOT NULL,
    tool_wear_min REAL NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
);

-- Prediction results
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    sensor_reading_id INTEGER NOT NULL,
    machine_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    failure_probability REAL NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    threshold_used REAL NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    recommendation TEXT,
    actual_failure BOOLEAN,
    was_correct BOOLEAN,
    FOREIGN KEY (sensor_reading_id) REFERENCES sensor_readings(id),
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
);

-- Failure events
CREATE TABLE failure_events (
    id INTEGER PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    failure_type VARCHAR(20) NOT NULL,
    occurred_at TIMESTAMP NOT NULL,
    description TEXT,
    downtime_hrs REAL,
    repair_cost REAL,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
);

-- Maintenance logs
CREATE TABLE maintenance_logs (
    id INTEGER PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triggered_by VARCHAR(20) DEFAULT 'manual',
    notes TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
);
```

### Database Configuration

**Development (SQLite):**
```env
DATABASE_URL=sqlite:///data/predictive_maintenance.db
```

**Production (PostgreSQL):**
```env
DATABASE_URL=postgresql://pdm_user:pdm_pass@localhost:5432/pdm_db
```

### Database Endpoints

#### GET /machines
List all registered machines with their details.

#### GET /machines/{machine_id}/summary
Full machine summary including readings, predictions, failures, and maintenance history.

#### GET /machines/{machine_id}/history
Sensor reading history for a specific machine.

#### GET /predictions/stats
Aggregate statistics for dashboard:
- Total predictions count
- Risk level distribution (High/Medium/Low)
- Average failure probability

#### GET /predictions/recent?limit=20
Most recent predictions across all machines.

#### GET /predictions/high-risk?hours=24
High-risk predictions within specified time window.

#### GET /predictions/trend?days=7
Daily risk level counts for trend analysis (dashboard charts).

### Data Flow

1. **Prediction Request** → API validates input → ML model inference
2. **Database Storage** → Machine registration → Sensor reading → Prediction result
3. **Dashboard Queries** → Real-time statistics → Historical trends → Risk alerts

---

## Model Performance

The model was evaluated using cross-validation and a hold-out test set.

| Metric | Validation | Test |
|--------|-----------|------|
| ROC-AUC | 0.9743 | 0.9784 |
| Avg Precision | 0.8514 | 0.8572 |
| F2 Score | 0.7843 | 0.7692 |

The threshold was optimized for **F2-score**, prioritizing recall to reduce the risk of missed failures in industrial environments.

## Robustness Evaluation

The model stability was evaluated using bootstrap resampling on the test set:

- F2 mean: 0.7673  
- Std: 0.0317  
- 95% CI: [0.7080, 0.8367]  

Result: The model shows acceptable stability with minor sensitivity to data distribution shifts.

---

## ML Engineering Decisions

| Decision | Rationale |
|---|---|
| XGBoost over deep learning | Superior on tabular sensor data; interpretable; fast |
| SMOTE on train set only | Prevents data leakage into validation/test |
| F2-optimized threshold | In PdM, false negatives (missed failures) cost more than false alarms |
| Shared features.py | Single source of truth prevents training-serving skew |
| Scaler saved separately | Allows version-independent updates |
| JSON structured logging | Enables log aggregation, alerting, SLA monitoring |
| Non-root Docker user | Production security hardening |
| Multi-stage Docker build | ~60% smaller final image |
| SQLAlchemy ORM | Database-agnostic codebase (SQLite dev, PostgreSQL prod) |
| Automatic persistence | Every prediction saved for audit trail and dashboard analytics |
| Foreign key relationships | Data integrity across machines, readings, and predictions |

---

## Risk Level Thresholds

| Risk Level | Probability Range | Action |
|---|---|---|
| 🟢 Low    | 0% – 35%  | Normal monitoring |
| 🟡 Medium | 35% – 65% | Monitor closely; flag for next maintenance |
| 🔴 High   | 65% – 100%| Immediate inspection within 24 hours |

---

## Extending the System

**Add a new feature:** Edit `model/features.py` → retrain → redeploy API. No other files change.

**Swap the model:** Replace XGBClassifier in `train.py` with any sklearn-compatible estimator. The rest of the pipeline is model-agnostic.

**Add SHAP explanations:** Replace `_build_risk_factors()` in `predictor.py` with `shap.TreeExplainer` for per-sample feature attribution.

**Add model versioning:** Prefix artifacts with a version hash (`model_v2_abc123.pkl`) and update `MODEL_VERSION` in `predictor.py`.

---

## License

MIT
