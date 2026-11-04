"""
Pytest unit tests for src.data.preprocess module.
"""

import os
import pandas as pd
import pytest

from src.data.preprocess import (
    load_data,
    clean_data,
    split_data,
    TARGET_COLUMN
)


@pytest.fixture
def raw_dataset_path():
    path = "data/raw/dataset.csv"
    assert os.path.exists(path), f"Raw dataset missing at {path}"
    return path


def test_load_data(raw_dataset_path):
    """Test raw dataset loading and dimensions."""
    df = load_data(raw_dataset_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 7000
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_clean_data(raw_dataset_path):
    """Test data cleaning: customerID dropped, TotalCharges numeric, target mapped."""
    df_raw = load_data(raw_dataset_path)
    df_clean = clean_data(df_raw)

    # 1. customerID dropped
    assert "customerID" not in df_clean.columns

    # 2. TotalCharges converted to numeric without nulls
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
    assert df_clean["TotalCharges"].isnull().sum() == 0

    # 3. Target is binary 0 and 1
    assert set(df_clean[TARGET_COLUMN].unique()).issubset({0, 1})

    # Exactly 11 rows with blank TotalCharges dropped from 7043 -> 7032
    assert df_clean.shape[0] == 7032


def test_split_data(raw_dataset_path):
    """Test train/test split stratification and sizes."""
    df_raw = load_data(raw_dataset_path)
    df_clean = clean_data(df_raw)

    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

    total_rows = len(df_clean)
    assert len(X_train) + len(X_test) == total_rows
    assert abs(len(X_test) / total_rows - 0.2) < 0.01

    # Verify stratified target proportions match
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.02


def test_processed_files_exist():
    """Verify preprocessed files are present and non-empty."""
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    assert os.path.exists(train_path), "train.csv missing"
    assert os.path.exists(test_path), "test.csv missing"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0
