"""
Module: src.models.evaluate
Responsibility: Evaluation functions and metrics reporting for Customer Churn models.
- Computes Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Plots and saves Confusion Matrix and ROC Curve
- Logs evaluation artifacts to MLflow
"""

import os
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)
import mlflow

from src.utils.logger import get_logger

logger = get_logger("model_evaluation")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes all standard binary classification metrics:
    Accuracy, Precision, Recall, F1-score, ROC-AUC.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_pred_proba is not None:
        try:
            # Handle 2D probability array or 1D positive class array
            if y_pred_proba.ndim == 2:
                prob_positive = y_pred_proba[:, 1]
            else:
                prob_positive = y_pred_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_positive))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str = "logs/confusion_matrix.png"
) -> str:
    """
    Plots and saves the confusion matrix for model predictions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to {output_path}")
    return output_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    output_path: str = "logs/roc_curve.png"
) -> Optional[str]:
    """
    Plots and saves the ROC curve for positive class probabilities.
    """
    if y_pred_proba is None:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if y_pred_proba.ndim == 2:
        prob_positive = y_pred_proba[:, 1]
    else:
        prob_positive = y_pred_proba

    fpr, tpr, _ = roc_curve(y_true, prob_positive)
    auc_score = roc_auc_score(y_true, prob_positive)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(f"ROC Curve: {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved ROC curve plot to {output_path}")
    return output_path


def log_evaluation_to_mlflow(
    metrics: Dict[str, float],
    cm_path: Optional[str] = None,
    roc_path: Optional[str] = None
) -> None:
    """
    Logs evaluation metrics and visual artifacts to active MLflow run.
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    if cm_path and os.path.exists(cm_path):
        mlflow.log_artifact(cm_path, artifact_path="plots")

    if roc_path and os.path.exists(roc_path):
        mlflow.log_artifact(roc_path, artifact_path="plots")
