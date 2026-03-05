"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction service.

Run with:
    pytest tests/test_api.py -v

Tests use TestClient (no live server needed).
Model artifacts must exist (run training pipeline first).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import predictor

client = TestClient(app)

# ── Load model artifacts once at test collection time ────────────────────────
@pytest.fixture(scope="session", autouse=True)
def load_model_artifacts():
    """Ensure model artifacts are loaded before any tests run."""
    predictor.load_artifacts()
    yield
    # Cleanup if needed

# ── Shared test fixture ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "machine_id":           "MIL-TEST-01",
    "product_type":         "M",
    "air_temp_c":           25.1,
    "process_temp_c":       36.4,
    "rotational_speed_rpm": 1551.0,
    "torque_nm":            42.8,
    "tool_wear_min":        108.0,
}

HIGH_RISK_PAYLOAD = {
    "machine_id":           "MIL-STRESS-99",
    "product_type":         "L",
    "air_temp_c":           28.5,
    "process_temp_c":       40.1,
    "rotational_speed_rpm": 1200.0,   # Low speed + high torque = overstrain
    "torque_nm":            68.0,     # High torque — well above nominal 40Nm
    "tool_wear_min":        250.0,    # Critical tool wear
}


# ── Health endpoint ────────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status"        in data
        assert "model_loaded"  in data
        assert "model_version" in data

    def test_health_status_is_string(self):
        response = client.get("/health")
        assert isinstance(response.json()["status"], str)


# ── Prediction endpoint — happy path ──────────────────────────────────────────
class TestPredictFailureEndpoint:

    def test_predict_returns_200(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        data = response.json()
        required_fields = [
            "failure_probability", "risk_level", "threshold_used",
            "recommendation", "top_risk_factors", "model_version",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_probability_is_valid_range(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        prob = response.json()["failure_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1] range"

    def test_predict_risk_level_valid_values(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        risk = response.json()["risk_level"]
        assert risk in ["Low", "Medium", "High"], f"Unexpected risk level: {risk}"

    def test_predict_machine_id_echoed(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        assert response.json()["machine_id"] == "MIL-TEST-01"

    def test_predict_top_risk_factors_is_list(self):
        response = client.post("/predict_failure", json=VALID_PAYLOAD)
        factors = response.json()["top_risk_factors"]
        assert isinstance(factors, list)
        assert len(factors) > 0

    def test_predict_without_machine_id(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "machine_id"}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 200
        assert response.json()["machine_id"] is None

    def test_high_risk_machine_scores_higher(self):
        """A stressed machine should produce a higher probability than a healthy one."""
        normal_prob = client.post("/predict_failure", json=VALID_PAYLOAD).json()["failure_probability"]
        high_prob   = client.post("/predict_failure", json=HIGH_RISK_PAYLOAD).json()["failure_probability"]
        assert high_prob > normal_prob, (
            f"Expected stressed machine ({high_prob}) > healthy machine ({normal_prob})"
        )


# ── Prediction endpoint — validation ──────────────────────────────────────────
class TestPredictValidation:

    def test_missing_required_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "air_temp_c"}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 422

    def test_torque_above_max_returns_422(self):
        payload = {**VALID_PAYLOAD, "torque_nm": 999.0}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 422

    def test_negative_tool_wear_returns_422(self):
        payload = {**VALID_PAYLOAD, "tool_wear_min": -10.0}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 422

    def test_invalid_product_type_returns_422(self):
        payload = {**VALID_PAYLOAD, "product_type": "X"}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 422

    def test_rpm_above_max_returns_422(self):
        payload = {**VALID_PAYLOAD, "rotational_speed_rpm": 9999.0}
        response = client.post("/predict_failure", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict_failure", json={})
        assert response.status_code == 422