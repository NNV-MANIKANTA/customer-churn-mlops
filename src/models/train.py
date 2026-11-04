"""
Module: src.models.train
Responsibility: Machine Learning training module with MLflow experiment tracking.
- Trains 3 candidate models: Logistic Regression, Decision Tree, Random Forest
- Evaluates models on test set using src.models.evaluate
- Tracks params, metrics, plots, and models in MLflow
- Compares models and selects the winner based on F1-score
- Bundles fitted preprocessor and winner into a complete end-to-end Pipeline
- Serializes complete pipeline to artifacts/model/model.joblib
"""

import os
from typing import Dict, Any, Tuple
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

from src.features.build_features import TARGET_COLUMN
from src.models.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    log_evaluation_to_mlflow
)
from src.utils.logger import get_logger

logger = get_logger("model_training")


def load_config(config_path: str = "params.yaml") -> dict:
    """Load configuration from params.yaml with fallback defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config file {config_path} not found. Using defaults.")
    return {}


def load_data(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Loads processed train and test datasets and splits into X and y."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed datasets missing. Run src.data.preprocess first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    logger.info(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def get_models(params: dict) -> Dict[str, Any]:
    """Instantiate models using hyperparameters from params.yaml."""
    lr_params = params.get("logistic_regression", {})
    dt_params = params.get("decision_tree", {})
    rf_params = params.get("random_forest", {})

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=lr_params.get("max_iter", 1000),
            C=lr_params.get("C", 1.0),
            solver=lr_params.get("solver", "lbfgs"),
            random_state=42
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=dt_params.get("max_depth", 5),
            min_samples_split=dt_params.get("min_samples_split", 10),
            random_state=dt_params.get("random_state", 42)
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 10),
            min_samples_split=rf_params.get("min_samples_split", 5),
            random_state=rf_params.get("random_state", 42)
        )
    }
    return models


def train_and_evaluate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "customer-churn-experiment"
) -> Tuple[str, Any, Dict[str, Dict[str, float]]]:
    """
    Trains each model, logs runs to MLflow, compares metrics,
    and returns the best model selected by F1-score.
    """
    mlflow.set_experiment(experiment_name)
    all_results: Dict[str, Dict[str, float]] = {}
    best_model_name = ""
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        logger.info(f"--- Training: {name} ---")
        with mlflow.start_run(run_name=name):
            # 1. Log parameters
            model_params = model.get_params()
            # Filter primitive serializable params
            clean_params = {k: v for k, v in model_params.items() if isinstance(v, (int, float, str, bool))}
            mlflow.log_params(clean_params)
            mlflow.set_tag("model_family", name)

            # 2. Train model
            model.fit(X_train, y_train)

            # 3. Predict on test set
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)

            # 4. Calculate metrics
            metrics = calculate_metrics(y_test.values, y_pred, y_pred_proba)
            all_results[name] = metrics

            logger.info(
                f"{name} Results -> Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1-Score: {metrics['f1_score']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            # 5. Generate and save diagnostic plots
            cm_path = plot_confusion_matrix(
                y_test.values, y_pred, name, output_path=f"logs/cm_{name}.png"
            )
            roc_path = plot_roc_curve(
                y_test.values, y_pred_proba, name, output_path=f"logs/roc_{name}.png"
            )

            # 6. Log metrics, plots, and model artifact to MLflow
            log_evaluation_to_mlflow(metrics, cm_path=cm_path, roc_path=roc_path)
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    serialization_format="cloudpickle"
                )
            except Exception as e:
                logger.warning(f"Could not log sklearn model with default format: {e}. Falling back to artifact log.")
                joblib.dump(model, f"logs/{name}.joblib")
                mlflow.log_artifact(f"logs/{name}.joblib", artifact_path="model")

            # Check if this model is the new leader by F1-score
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model_name = name
                best_model = model

    logger.info(f"[BEST MODEL] {best_model_name} (F1-score = {best_f1:.4f})")
    return best_model_name, best_model, all_results


def save_full_pipeline(
    best_model: Any,
    preprocessor_path: str = "artifacts/model/preprocessor.joblib",
    output_model_path: str = "artifacts/model/model.joblib"
) -> Pipeline:
    """
    Assembles fitted preprocessor + best classifier into a complete
    end-to-end Pipeline and saves it to artifacts/model/model.joblib.
    """
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)
    if hasattr(preprocessor, "set_output"):
        preprocessor.set_output(transform="pandas")

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_model)
        ]
    )

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(full_pipeline, output_model_path)
    logger.info(f"Saved complete end-to-end ML pipeline to {output_model_path}")
    return full_pipeline


def run_training(config_path: str = "params.yaml") -> None:
    """Orchestrates model training, evaluation, comparison, and export."""
    logger.info("========== Starting ML Model Training ==========")
    params = load_config(config_path)

    data_cfg = params.get("data", {})
    train_path = data_cfg.get("train_path", "data/processed/train.csv")
    test_path = data_cfg.get("test_path", "data/processed/test.csv")

    model_cfg = params.get("model", {})
    artifact_path = model_cfg.get("artifact_path", "artifacts/model/model.joblib")
    experiment_name = model_cfg.get("experiment_name", "customer-churn-experiment")

    # 1. Load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # 2. Instantiate models
    models = get_models(params)

    # 3. Train, evaluate & log runs in MLflow
    best_name, best_model, results = train_and_evaluate_models(
        models, X_train, y_train, X_test, y_test, experiment_name=experiment_name
    )

    # 4. Print comparison summary table
    print("\n" + "=" * 75)
    print(f"{'MODEL COMPARISON TABLE':^75}")
    print("=" * 75)
    print(f"{'Model Name':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 75)
    for m_name, m_res in results.items():
        leader_marker = " [BEST]" if m_name == best_name else ""
        print(
            f"{m_name + leader_marker:<22} | "
            f"{m_res['accuracy']:<9.4f} | "
            f"{m_res['precision']:<9.4f} | "
            f"{m_res['recall']:<9.4f} | "
            f"{m_res['f1_score']:<9.4f} | "
            f"{m_res['roc_auc']:<9.4f}"
        )
    print("=" * 75)
    print(f"Winner selected by F1-Score: {best_name}\n")

    # 5. Save the complete pipeline
    save_full_pipeline(best_model, output_model_path=artifact_path)

    # 6. Save best metrics for DVC metrics tracking
    import json
    metrics_summary = {
        "best_model": best_name,
        "accuracy": results[best_name]["accuracy"],
        "precision": results[best_name]["precision"],
        "recall": results[best_name]["recall"],
        "f1_score": results[best_name]["f1_score"],
        "roc_auc": results[best_name]["roc_auc"]
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info("Saved metrics.json for DVC tracking.")
    logger.info("========== Model Training Stage Completed Successfully ==========")


if __name__ == "__main__":
    run_training()
