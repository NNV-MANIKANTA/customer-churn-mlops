"""
Pytest integration tests for FastAPI backend endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """Test GET / endpoint returns 200 OK and expected keys."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_check():
    """Test GET /health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "pipeline_type" in data


def test_predict_endpoint_high_risk():
    """Test POST /predict with a customer likely to churn (month-to-month, high charges)."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 85.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert "churn_label" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["High", "Medium", "Low"]
    assert "retention_recommendation" in data


def test_predict_endpoint_low_risk():
    """Test POST /predict with a customer unlikely to churn (2-year contract, high tenure)."""
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] == 0
    assert data["churn_label"] == "No"
    assert data["churn_probability"] < 0.5


def test_predict_invalid_input():
    """Test POST /predict with missing required fields returns 422 Unprocessable Entity."""
    bad_payload = {
        "tenure": 10
        # Missing all other required features
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_model_info():
    """Test GET /model-info returns 200 with model metrics and metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "Customer Churn Prediction MLOps"
    assert "best_model" in data
    assert "metrics" in data
    assert data["feature_count"] == 19
