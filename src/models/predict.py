"""
Module: src.models.predict
Responsibility: Production inference service for Telco Customer Churn.
- Loads complete end-to-end Pipeline (preprocessor + model) from artifacts/model/model.joblib
- Accepts raw customer features (dict, json, or DataFrame)
- Predicts churn label (0: No Churn, 1: Churn) and churn probability
"""

import os
from typing import Dict, Any, Union, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger("model_prediction")

DEFAULT_MODEL_PATH = "artifacts/model/model.joblib"
_cached_pipeline: Union[Pipeline, None] = None


def get_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """
    Loads and caches the end-to-end model pipeline from disk.
    """
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model pipeline artifact not found at '{model_path}'. "
                "Ensure src.models.train has been executed."
            )
        logger.info(f"Loading model pipeline artifact from: {model_path}")
        _cached_pipeline = joblib.load(model_path)
    return _cached_pipeline


def predict_churn(
    customer_data: Union[Dict[str, Any], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH
) -> Tuple[int, float, str]:
    """
    Runs end-to-end inference on raw customer attributes.

    Args:
        customer_data: Dict or DataFrame containing raw customer features.
        model_path: Path to serialized model.joblib artifact.

    Returns:
        Tuple:
            - prediction (int): 0 (retained) or 1 (churn)
            - probability (float): probability of churn [0.0, 1.0]
            - label (str): 'No' or 'Yes'
    """
    pipeline = get_pipeline(model_path)

    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()
    else:
        raise ValueError("customer_data must be a dict or a pandas DataFrame.")

    # Ensure numerical types are properly cast
    if "TotalCharges" in df_input.columns:
        df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce").fillna(0.0)
    if "MonthlyCharges" in df_input.columns:
        df_input["MonthlyCharges"] = pd.to_numeric(df_input["MonthlyCharges"], errors="coerce").fillna(0.0)
    if "tenure" in df_input.columns:
        df_input["tenure"] = pd.to_numeric(df_input["tenure"], errors="coerce").fillna(0)

    # Predict label & probability
    pred = int(pipeline.predict(df_input)[0])

    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(df_input)[0, 1])
    else:
        prob = float(pred)

    label = "Yes" if pred == 1 else "No"
    return pred, prob, label


if __name__ == "__main__":
    # Smoke test inference with a sample customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    prediction, probability, label = predict_churn(sample_customer)
    print(f"\n--- Inference Smoke Test ---")
    print(f"Prediction  : {prediction} ({label})")
    print(f"Probability : {probability:.4f} ({probability * 100:.2f}%)")
