"""
Pytest unit tests for src.models.predict module and model inference artifact.
"""

import os
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.predict import get_pipeline, predict_churn, DEFAULT_MODEL_PATH


def test_model_artifact_exists():
    """Ensure the trained model pipeline artifact is serialized on disk."""
    assert os.path.exists(DEFAULT_MODEL_PATH), f"Model missing at {DEFAULT_MODEL_PATH}"


def test_load_pipeline():
    """Verify that the loaded artifact is a valid Scikit-learn Pipeline with expected steps."""
    pipeline = get_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_predict_churn_high_risk_customer():
    """Verify inference for a high-risk churn candidate."""
    high_risk_data = {
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
        "MonthlyCharges": 85.50,
        "TotalCharges": 85.50
    }
    prediction, probability, label = predict_churn(high_risk_data)
    assert prediction in [0, 1]
    assert label in ["No", "Yes"]
    assert 0.0 <= probability <= 1.0
    # Customer with tenure=1 and fiber optic + month-to-month has high churn probability
    assert probability > 0.5


def test_predict_churn_loyal_customer():
    """Verify inference for a long-tenure customer with 2-year contract."""
    loyal_data = {
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
    prediction, probability, label = predict_churn(loyal_data)
    assert prediction == 0
    assert label == "No"
    assert probability < 0.25

