"""
Module: src.data.preprocess
Responsibility: Production data preprocessing for Telco Customer Churn.
- Loads raw dataset
- Cleans data (converts TotalCharges to numeric, handles missing values, removes customerID)
- Maps binary target (Churn: Yes -> 1, No -> 0)
- Splits into stratified Train and Test sets BEFORE transformations to prevent data leakage
- Fits Scikit-learn ColumnTransformer on Train only, transforms both Train and Test
- Exports processed datasets to data/processed/
"""

import os
from typing import Tuple, Optional
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from src.features.build_features import (
    get_feature_pipeline,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.utils.logger import get_logger

logger = get_logger("data_preprocessing")


def load_params(params_path: str = "params.yaml") -> dict:
    """
    Load configuration parameters from YAML file.
    Falls back to safe defaults if file is not found.
    """
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {params_path}")
            return params
    logger.warning(f"Config {params_path} not found. Using default parameters.")
    return {
        "data": {
            "raw_path": "data/raw/dataset.csv",
            "train_path": "data/processed/train.csv",
            "test_path": "data/processed/test.csv",
            "test_size": 0.2,
            "random_state": 42
        }
    }


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset from {file_path} with shape {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning:
    1. Drop customerID column (no predictive value).
    2. Convert TotalCharges to numeric, coercing spaces to NaN.
    3. Drop records with NaN TotalCharges (new customers with tenure=0).
    4. Encode target variable Churn to binary (Yes: 1, No: 0).
    """
    df = df.copy()

    # 1. Remove customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped 'customerID' column.")

    # 2. Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        null_count = df["TotalCharges"].isnull().sum()
        if null_count > 0:
            logger.info(f"Detected {null_count} missing values in 'TotalCharges'. Dropping rows...")
            df = df.dropna(subset=["TotalCharges"])

    # 3. Encode target variable
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype == "object":
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
            logger.info("Encoded target 'Churn' to binary integers (Yes: 1, No: 0).")

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and test sets using stratified sampling.
    Crucial: Done BEFORE transformation to strictly prevent data leakage.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        f"Data split successfully. Train features: {X_train.shape}, "
        f"Test features: {X_test.shape} (Stratified Churn Rate: {y_train.mean():.3f})"
    )
    return X_train, X_test, y_train, y_test


def transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Applies Scikit-learn Pipeline (ColumnTransformer):
    - Fits StandardScaler and OneHotEncoder on Train data ONLY.
    - Transforms both Train and Test data.
    - Preserves transformed feature names.
    """
    preprocessor = get_feature_pipeline()

    # Fit on train ONLY, then transform train
    X_train_transformed = preprocessor.fit_transform(X_train)
    # Transform test (NEVER fit on test)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract clean feature names
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    logger.info(f"Transformed features into {len(feature_names)} columns after one-hot encoding.")
    return X_train_df, X_test_df, preprocessor


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: str,
    test_path: str,
    preprocessor: Optional[ColumnTransformer] = None,
    preprocessor_path: Optional[str] = "artifacts/model/preprocessor.joblib"
) -> None:
    """
    Save processed training and testing DataFrames to CSV, and optionally save
    the fitted preprocessor pipeline artifact.
    """
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved processed train data to {train_path} ({train_df.shape})")
    logger.info(f"Saved processed test data to {test_path} ({test_df.shape})")

    if preprocessor and preprocessor_path:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Saved fitted preprocessor pipeline to {preprocessor_path}")


def run_preprocessing(params_path: str = "params.yaml") -> None:
    """
    Full orchestrator for the preprocessing stage.
    """
    logger.info("========== Starting Preprocessing Stage ==========")
    params = load_params(params_path)
    data_cfg = params.get("data", {})

    raw_path = data_cfg.get("raw_path", "data/raw/dataset.csv")
    train_path = data_cfg.get("train_path", "data/processed/train.csv")
    test_path = data_cfg.get("test_path", "data/processed/test.csv")
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)

    # 1. Load data
    df = load_data(raw_path)

    # 2. Clean data
    df_clean = clean_data(df)

    # 3. Train-test split (Prevents data leakage)
    X_train, X_test, y_train, y_test = split_data(
        df_clean, test_size=test_size, random_state=random_state
    )

    # 4. Feature transformations via Scikit-learn Pipeline
    X_train_proc, X_test_proc, preprocessor = transform_features(X_train, X_test)

    # 5. Combine features with target for storage
    train_processed = X_train_proc.copy()
    train_processed[TARGET_COLUMN] = y_train.values

    test_processed = X_test_proc.copy()
    test_processed[TARGET_COLUMN] = y_test.values

    # 6. Save datasets
    save_processed_data(
        train_processed,
        test_processed,
        train_path=train_path,
        test_path=test_path,
        preprocessor=preprocessor
    )
    logger.info("========== Preprocessing Stage Completed Successfully ==========")


if __name__ == "__main__":
    run_preprocessing()
