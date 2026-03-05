"""
app/schemas.py
--------------
Pydantic models for the AI4I 2020 Predictive Maintenance API.

Input fields match the AI4I 2020 dataset:
  - Stephan Matzka, HTW Berlin, Germany
  - DOI: https://doi.org/10.24432/C5HS5C
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


class ProductType(str, Enum):
    LOW    = "L"   # 50% of products — lowest quality variant
    MEDIUM = "M"   # 30% of products
    HIGH   = "H"   # 20% of products — highest quality variant


class SensorReading(BaseModel):
    """
    Input: one observation for a single milling machine.
    Based on the AI4I 2020 dataset schema (HTW Berlin / UCI).
    """
    machine_id:           Optional[str]  = None
    product_type:         ProductType    = Field(..., description="Product quality variant: L=Low, M=Medium, H=High.")
    air_temp_c:           float          = Field(..., ge=-50.0,  le=100.0,  description="Air temperature in Celsius (converted from Kelvin).")
    process_temp_c:       float          = Field(..., ge=-50.0,  le=150.0,  description="Process temperature in Celsius.")
    rotational_speed_rpm: float          = Field(..., ge=0.0,    le=3000.0, description="Rotational speed in RPM.")
    torque_nm:            float          = Field(..., ge=0.0,    le=100.0,  description="Torque in Newton-meters.")
    tool_wear_min:        float          = Field(..., ge=0.0,    le=300.0,  description="Tool wear accumulation in minutes.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_id":           "MIL-0042",
                "product_type":         "M",
                "air_temp_c":           25.1,
                "process_temp_c":       36.4,
                "rotational_speed_rpm": 1551.0,
                "torque_nm":            42.8,
                "tool_wear_min":        108.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction output from POST /predict_failure."""
    machine_id:          Optional[str]
    failure_probability: float
    risk_level:          RiskLevel
    threshold_used:      float
    recommendation:      str
    top_risk_factors:    List[str]
    model_version:       str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str