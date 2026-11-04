"""
Feature engineering definitions and ColumnTransformer pipeline for Telco Churn.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Canonical feature lists
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

TARGET_COLUMN: str = "Churn"


def get_feature_pipeline() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer that:
    1. Scales numerical features using StandardScaler.
    2. Encodes categorical features using OneHotEncoder (handle_unknown='ignore').
    
    Returns:
        ColumnTransformer: Unfitted preprocessor pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor
