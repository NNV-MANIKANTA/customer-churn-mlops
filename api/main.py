"""
FastAPI Backend Service for Telco Customer Churn Prediction.

Endpoints:
  GET  /            - Welcome & service metadata
  GET  /health      - Service health check & model readiness
  POST /predict     - End-to-end churn prediction & probability
  GET  /model-info  - Model metadata, metrics, and tracking info
"""

import os
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.models.predict import get_pipeline, predict_churn
from src.utils.logger import get_logger

logger = get_logger("fastapi_backend")


# -------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------
class CustomerInput(BaseModel):
    gender: str = Field(..., description="Customer gender ('Male' or 'Female')", json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (1, 0)", json_schema_extra={"example": 0})
    Partner: str = Field(..., description="Whether customer has a partner ('Yes', 'No')", json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., description="Whether customer has dependents ('Yes', 'No')", json_schema_extra={"example": "No"})
    tenure: int = Field(..., ge=0, description="Months customer has stayed with company", json_schema_extra={"example": 1})
    PhoneService: str = Field(..., description="Whether customer has phone service ('Yes', 'No')", json_schema_extra={"example": "No"})
    MultipleLines: str = Field(..., description="'No phone service', 'No', or 'Yes'", json_schema_extra={"example": "No phone service"})
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', or 'No'", json_schema_extra={"example": "DSL"})
    OnlineSecurity: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "No"})
    StreamingMovies: str = Field(..., description="'Yes', 'No', or 'No internet service'", json_schema_extra={"example": "No"})
    Contract: str = Field(..., description="'Month-to-month', 'One year', 'Two year'", json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'", json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'", json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly", json_schema_extra={"example": 29.85})
    TotalCharges: float = Field(..., ge=0.0, description="The total amount charged to the customer", json_schema_extra={"example": 29.85})


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 for No Churn, 1 for Churn")
    churn_label: str = Field(..., description="'No' or 'Yes'")
    churn_probability: float = Field(..., description="Predicted probability of churning (0.0 to 1.0)")
    risk_level: str = Field(..., description="'High', 'Medium', or 'Low'")
    retention_recommendation: str = Field(..., description="Automated business action recommendation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_type: str


class ModelInfoResponse(BaseModel):
    project_name: str
    best_model: str
    metrics: Dict[str, Any]
    feature_count: int
    tracking_framework: str


# -------------------------------------------------------------------------
# Lifespan context manager for warm model loading (No Retraining)
# -------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FastAPI Backend service...")
    try:
        # Load the pipeline artifact once into memory at startup
        pipeline = get_pipeline()
        logger.info(f"Model pipeline verified and cached successfully: {type(pipeline).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model pipeline at startup: {e}")
    yield
    logger.info("Shutting down FastAPI service.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST API serving the Telco Customer Churn Scikit-learn Pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get("/", summary="Root Welcome Endpoint")
def read_root():
    """Returns welcoming service metadata and available documentation."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Validates that the API is running and the trained pipeline is loaded."""
    try:
        pipeline = get_pipeline()
        model_ready = pipeline is not None
    except Exception:
        model_ready = False

    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        model_loaded=model_ready,
        pipeline_type="Scikit-learn Pipeline (ColumnTransformer + Classifier)"
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Customer Churn")
def predict(customer: CustomerInput):
    """
    Accepts raw customer attributes, runs through the production pipeline,
    and returns churn prediction, probability, and retention guidance.
    """
    try:
        customer_dict = customer.model_dump()
        prediction, probability, label = predict_churn(customer_dict)

        # Risk level determination
        if probability >= 0.65:
            risk = "High"
            recommendation = "Immediate retention action required: Offer long-term contract discount or dedicated tech support."
        elif probability >= 0.40:
            risk = "Medium"
            recommendation = "Proactive engagement: Check satisfaction with current plan and offer bundle promotions."
        else:
            risk = "Low"
            recommendation = "Customer satisfied: Good candidate for cross-selling value-added services."

        return PredictionResponse(
            churn_prediction=prediction,
            churn_label=label,
            churn_probability=round(probability, 4),
            risk_level=risk,
            retention_recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse, summary="Get Model Metadata & Metrics")
def get_model_info():
    """Returns best model architecture, evaluation metrics, and tracking metadata."""
    metrics_path = "metrics.json"
    metrics_data = {}
    best_model_name = "Logistic_Regression"

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                best_model_name = metrics_data.get("best_model", best_model_name)
        except Exception as e:
            logger.warning(f"Could not read metrics.json: {e}")

    return ModelInfoResponse(
        project_name="Customer Churn Prediction MLOps",
        best_model=best_model_name,
        metrics=metrics_data,
        feature_count=19,
        tracking_framework="MLflow & DVC"
    )
