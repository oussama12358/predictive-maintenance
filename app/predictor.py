"""
app/predictor.py
----------------
Handles artifact loading and single-sample inference.

Design principles:
  - Artifacts are loaded ONCE at startup (not per request) — critical for latency
  - Uses the shared features.py module to prevent training-serving skew
  - Returns structured result dict (not raw floats) for clean API layer
  - Risk level thresholds and recommendations are data-driven, not hardcoded strings
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from model.features import engineer_features, get_feature_columns
from app.schemas import RiskLevel


# ── Artifact paths ────────────────────────────────────────────────────────────
MODEL_PATH     = Path("model/artifacts/model.pkl")
SCALER_PATH    = Path("model/artifacts/scaler.pkl")
THRESHOLD_PATH = Path("model/artifacts/threshold.json")
MODEL_VERSION  = "xgb-v1.0"

# ── Risk level bands (probability ranges) ─────────────────────────────────────
RISK_BANDS = {
    RiskLevel.LOW:    (0.00, 0.35),
    RiskLevel.MEDIUM: (0.35, 0.65),
    RiskLevel.HIGH:   (0.65, 1.00),
}

# ── Maintenance recommendations per risk level ────────────────────────────────
RECOMMENDATIONS = {
    RiskLevel.LOW:    "No immediate action required. Continue normal monitoring schedule.",
    RiskLevel.MEDIUM: "Flag for next scheduled maintenance window. Monitor closely over next 48 hours.",
    RiskLevel.HIGH:   "Schedule immediate inspection within 24 hours. Consider temporary shutdown if vibration or temperature continues rising.",
}


class PredictiveMaintenancePredictor:
    """
    Encapsulates model loading and inference logic.
    Instantiated once at API startup and reused for all requests.
    """

    def __init__(self) -> None:
        self.model       = None
        self.scaler      = None
        self.threshold   = None
        self._is_loaded  = False

    def load_artifacts(self) -> None:
        """
        Load model, scaler, and threshold from disk.
        Called once during FastAPI startup event.

        Raises:
            FileNotFoundError: If any artifact is missing (training not yet run).
        """
        for path in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {path}. "
                    "Run 'python -m model.train' to generate artifacts first."
                )

        self.model   = joblib.load(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)

        with open(THRESHOLD_PATH, "r") as f:
            threshold_data = json.load(f)
        self.threshold = float(threshold_data["threshold"])

        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, sensor_data: dict) -> dict:
        """
        Run inference on a single AI4I sensor reading.

        Args:
            sensor_data: Dict matching SensorReading fields (machine_id excluded).
        Returns:
            Dict with failure_probability, risk_level, recommendation,
            top_risk_factors, threshold_used, model_version.
        """
        if not self._is_loaded:
            raise RuntimeError("Predictor not loaded. Call load_artifacts() first.")

        machine_id = sensor_data.pop("machine_id", None)

        # Encode product_type: L=0, M=1, H=2
        product_type_raw = sensor_data.pop("product_type", "M")
        type_map = {"L": 0, "M": 1, "H": 2}
        sensor_data["product_type_encoded"] = type_map.get(str(product_type_raw).upper(), 1)

        # Convert Celsius inputs (already in Celsius from API) — keep as-is
        # The load_ai4i.py converts K→C during training; API receives C directly

        # Build single-row DataFrame
        df = pd.DataFrame([sensor_data])

        # Apply feature engineering (same function as training)
        df = engineer_features(df)

        # Select and order features exactly as during training
        feature_cols = get_feature_columns()
        X = df[feature_cols].values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probability
        failure_prob = float(self.model.predict_proba(X_scaled)[0, 1])

        # Risk level
        risk_level = self._classify_risk(failure_prob)

        # Top risk factors
        top_risk_factors = self._build_risk_factors(df, feature_cols)

        return {
            "machine_id":          machine_id,
            "failure_probability": round(failure_prob, 4),
            "risk_level":          risk_level,
            "threshold_used":      round(self.threshold, 4),
            "recommendation":      RECOMMENDATIONS[risk_level],
            "top_risk_factors":    top_risk_factors,
            "model_version":       MODEL_VERSION,
        }

    def _classify_risk(self, probability: float) -> RiskLevel:
        """Map continuous probability to categorical risk level."""
        for level, (low, high) in RISK_BANDS.items():
            if low <= probability < high:
                return level
        return RiskLevel.HIGH  # Edge case: probability == 1.0

    def _build_risk_factors(self, df: pd.DataFrame, feature_cols: list) -> list:
        """
        Returns human-readable descriptions of the top 3 features
        ranked by model feature importance.

        For production, replace with SHAP values for per-sample explanations.
        """
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]

        factors = []
        for idx in top_indices:
            col   = feature_cols[idx]
            value = df[col].iloc[0]
            importance = importances[idx]
            factors.append(f"{col}: {value:.3f}  (importance: {importance:.3f})")

        return factors


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiated once; loaded during FastAPI startup event in main.py
predictor = PredictiveMaintenancePredictor()