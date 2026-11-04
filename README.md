# Customer Churn Prediction – End-to-End MLOps Project

An end-to-end Machine Learning and MLOps project that predicts customer churn using the IBM Telco Customer Churn dataset. The project covers the complete ML lifecycle, including exploratory data analysis, data preprocessing, feature engineering, model training, experiment tracking, data versioning, REST API development, and an interactive Streamlit application.

The objective is to identify customers who are likely to leave a telecom service and provide actionable retention recommendations.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Folder Structure](#project-folder-structure)
- [Dataset Description](#dataset-description)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Streamlit Application](#streamlit-application)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Data and Pipeline Versioning with DVC](#data-and-pipeline-versioning-with-dvc)
- [Testing](#testing)
- [Docker Support](#docker-support)
- [Future Improvements](#future-improvements)
- [Author & License](#author)

---

## Project Overview
Customer churn is a major challenge for subscription-based businesses. Acquiring new customers is often more expensive than retaining existing ones.

This project uses machine learning classification algorithms to predict whether a telecom customer is likely to discontinue their service.

The trained model is integrated with a **FastAPI backend** and a **Streamlit frontend**, allowing users to enter customer information and receive:
- **Churn prediction** (Yes/No)
- **Churn probability**
- **Customer risk level** (Low, Medium, High)
- **Retention recommendations**

The project follows an MLOps-oriented development approach to organize, track, test, and serve machine learning models.

---

## Key Features
- **Exploratory Data Analysis (EDA)** with distribution charts and correlation analysis.
- **Data Cleaning and Preprocessing** preventing data leakage with Scikit-learn Pipelines.
- **Feature Engineering & One-Hot Encoding** using `ColumnTransformer`.
- **Multiple Machine Learning Models** (Logistic Regression, Decision Tree, Random Forest).
- **Model Evaluation and Comparison** using Precision, Recall, F1-Score, and ROC-AUC.
- **MLflow Experiment Tracking** logging parameters, metrics, plots, and models.
- **DVC Data and Pipeline Versioning** managing dataset hashes, pipeline stages, and `dvc repro`.
- **FastAPI REST API** with Pydantic request validation and warm model caching.
- **Interactive Streamlit Frontend** featuring demo customer presets, health checks, and risk badges.
- **Automated Pytest Suite** validating endpoints, input schemas, and data pipelines.
- **Docker Containerization** with multi-container `docker-compose` orchestration.
- **GitHub Actions CI Workflow** automating build and test on push.

---

## System Architecture

```text
               IBM Telco Churn Dataset
                         │
                         ▼
             Exploratory Data Analysis
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                   Model Training
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Logistic        Decision        Random
    Regression         Tree          Forest
          └──────────────┬──────────────┘
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
               Best Model Selection
          ┌──────────────┴──────────────┐
          ▼                             ▼
   MLflow Tracking                 DVC Pipeline
          └──────────────┬──────────────┘
                         │
                         ▼
                 Saved ML Pipeline
            (artifacts/model/model.joblib)
                         │
                         ▼
                  FastAPI Backend
                    (Port 8000)
                         │
                         ▼
                Streamlit Frontend
                    (Port 8501)
                         │
                         ▼
              Churn Prediction Results
```

---

## End-to-End Workflow

1. **Data Collection:** Load the IBM Telco Customer Churn dataset into `data/raw/dataset.csv`. Target variable: `Churn`.
2. **Exploratory Data Analysis:** Analyze numerical and categorical distributions, missing values, class imbalances, and key risk factors in `notebooks/eda.ipynb`.
3. **Data Preprocessing:** Handle missing `TotalCharges`, drop `customerID`, map target to binary (0/1), and split train/test with stratification to avoid data leakage (`src/data/preprocess.py`).
4. **Feature Engineering:** Apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features using Scikit-learn's `ColumnTransformer` (`src/features/build_features.py`).
5. **Model Training:** Train and compare Logistic Regression, Decision Tree, and Random Forest models (`src/models/train.py`).
6. **Model Evaluation:** Calculate Accuracy, Precision, Recall, F1-Score, and ROC-AUC; plot Confusion Matrix and ROC Curve (`src/models/evaluate.py`).
7. **Experiment Tracking:** Use MLflow to record parameters, evaluation metrics, and artifacts under experiment `customer-churn-experiment`.
8. **Data & Pipeline Versioning:** Use DVC to track dataset hashes (`dataset.csv.dvc`) and reproduce pipeline stages via `dvc repro`.
9. **Model Serving:** Serve the complete preprocessor + model pipeline via FastAPI with Pydantic schemas (`api/main.py`).
10. **Frontend Integration:** Provide an interactive Streamlit UI for customer risk assessment (`frontend/app.py`).

---

## Project Folder Structure

```text
customer-churn-mlops/
│
├── .dvc/
│   ├── config
│   └── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── dataset.csv
│   │   └── dataset.csv.dvc
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── artifacts/
│   └── model/
│       ├── model.joblib
│       └── preprocessor.joblib
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       └── logger.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .dvcignore
├── .gitignore
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── params.yaml
├── requirements.txt
└── README.md
```

### Folder Responsibilities

| Folder / File | Responsibility |
| :--- | :--- |
| `api/` | FastAPI backend and prediction endpoints |
| `frontend/` | Streamlit user interface |
| `data/raw/` | Original dataset and DVC pointer (`.dvc`) |
| `data/processed/` | Processed training and testing datasets |
| `artifacts/model/` | Serialized model and preprocessor pipeline artifacts |
| `src/data/` | Data loading, cleaning, and preprocessing |
| `src/features/` | Feature engineering and `ColumnTransformer` definitions |
| `src/models/` | Training, evaluation, and inference modules |
| `src/utils/` | Centralized rotating logging configuration |
| `notebooks/` | Exploratory data analysis notebook |
| `tests/` | Automated unit and integration tests |
| `.github/workflows/` | GitHub Actions CI automation pipeline |
| `.dvc/` | DVC internal configuration |
| `dvc.yaml` | DVC pipeline DAG definition |
| `dvc.lock` | Pipeline dependency and output tracking hashes |
| `params.yaml` | Central hyperparameters configuration file |
| `metrics.json` | Latest model evaluation metrics |

> **Note:** Processed datasets and model artifacts are generated locally by the DVC pipeline and are excluded from Git commits via `.gitignore`.

---

## Dataset Description

* **Dataset:** IBM Telco Customer Churn
* **Records:** 7,043 customer accounts (7,032 after dropping 11 new customers with blank total charges)
* **Features:** 19 predictive customer attributes + 1 identifier (`customerID`)
* **Target Variable:** `Churn`

| Value | Meaning |
| :--- | :--- |
| **Yes** | Customer has churned (cancelled service) |
| **No** | Customer has not churned (retained) |

---

## Machine Learning Models

| Model | Purpose |
| :--- | :--- |
| **Logistic Regression** | Baseline linear classification model with calibrated probability outputs |
| **Decision Tree** | Interpretable rule-based non-linear classification |
| **Random Forest** | Ensemble bagging model with robust decision voting |

---

## Model Evaluation

The project evaluates models using multiple classification metrics rather than relying solely on accuracy due to the 26.5% class imbalance.

| Metric | Purpose |
| :--- | :--- |
| **Accuracy** | Overall prediction correctness |
| **Precision** | Correctness of positive churn predictions (minimizing false alarms) |
| **Recall** | Ability to capture actual churners (minimizing missed churners) |
| **F1-score** | Harmonic balance between precision and recall |
| **ROC-AUC** | Model ability to rank churn probabilities across all thresholds |

### Reported Experiment Metrics:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.8046 |
| **Precision** | 0.6496 |
| **Recall** | 0.5749 |
| **F1-Score** | **0.6099** |
| **ROC-AUC** | **0.8359** |

* **Best Model Selected:** **Logistic Regression** (selected by F1-Score).
* Metrics are automatically refreshed in `metrics.json` whenever the pipeline is rerun.

---

## Technologies Used

| Category | Technologies |
| :--- | :--- |
| **Programming Language** | Python 3.11 |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit |
| **HTTP Requests** | Requests, HTTPX |
| **Model Serialization** | Joblib |
| **Testing** | Pytest |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git, GitHub |

---

## Installation and Setup

### Prerequisites
* Python 3.11
* Git
* VS Code or another Python IDE
* Docker & Docker Compose (optional)

### 1. Clone the Repository
```bash
git clone https://github.com/NNV-MANIKANTA/customer-churn-mlops.git
cd customer-churn-mlops
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
Place the raw Telco Customer Churn CSV file at:
```text
data/raw/dataset.csv
```
If a DVC remote is configured, pull the tracked dataset:
```bash
dvc pull
```

### 5. Run the Complete MLOps Pipeline
```bash
dvc repro
```
Check pipeline status:
```bash
dvc status
```

---

## Running the Project

Run the backend and frontend in separate terminals:

### 1. Start FastAPI Backend
```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
* **Base URL:** `http://127.0.0.1:8000`
* **Swagger UI:** `http://127.0.0.1:8000/docs`

### 2. Start Streamlit Frontend
```bash
python -m streamlit run frontend/app.py --server.port 8501
```
* **Frontend URL:** `http://localhost:8501`

---

## API Documentation

FastAPI provides automatic interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### Available Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service information & endpoint directory |
| `GET` | `/health` | API and model pipeline readiness health check |
| `POST` | `/predict` | Predict customer churn and generate retention guidance |
| `GET` | `/model-info` | Best model metadata and latest evaluation metrics |

### Example Prediction Request (`POST /predict`)
```json
{
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
```

### Example Prediction Response
```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.7956,
  "risk_level": "High",
  "retention_recommendation": "Immediate retention action required: Offer long-term contract discount or dedicated tech support."
}
```

---

## Streamlit Application

The Streamlit frontend provides an intuitive web interface for customer risk assessment.

### Capabilities:
- Enter customer demographic, service, and contract attributes.
- Use **one-click demo profiles** (*High-Risk Churn Candidate* or *Loyal Customer*).
- Inspect live backend health and model metadata directly in the sidebar.
- Submit customer data to the FastAPI `/predict` backend.
- View real-time churn prediction, probability bar, and risk level.
- Review AI-generated retention recommendations.

The frontend communicates with the backend exclusively through HTTP requests without loading or retraining the ML model.

---

## Experiment Tracking with MLflow

MLflow tracks all model training runs, hyperparameters, and artifacts.

Start the MLflow UI:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Open your browser at: `http://127.0.0.1:5000`  
Experiment Name: `customer-churn-experiment`

---

## Data and Pipeline Versioning with DVC

DVC tracks data versions and orchestrates the execution DAG:
- `dvc.yaml`: Defines pipeline stages, dependencies, outputs, and metrics.
- `dvc.lock`: Records exact cryptographic hashes for reproducibility.
- `data/raw/dataset.csv.dvc`: Tracks the raw dataset.

### Useful Commands:
```bash
dvc status        # Check if pipeline components are modified
dvc dag           # Visualize pipeline dependency graph
dvc repro         # Reproduce and re-execute pipeline stages
dvc metrics show  # Display latest model evaluation metrics
```

---

## Testing

Pytest automates testing for API endpoints and data transformations.

Run the test suite:
```bash
python -m pytest -v
```

Run specific test modules:
```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_preprocessing.py -v
```

---

## Docker Support

Both backend and frontend services are containerized.

### Build Backend Image:
```bash
docker build -t customer-churn-api -f Dockerfile .
```

### Build Frontend Image:
```bash
docker build -t customer-churn-frontend -f Dockerfile.frontend .
```

### Run Multi-Container Application with Docker Compose:
```bash
docker compose up --build
```
Access the application at `http://localhost:8501`.

---

## Future Improvements
- Configure a shared cloud DVC remote (AWS S3 / GCP Storage).
- Deploy the FastAPI backend on AWS ECS / EC2.
- Deploy the Streamlit frontend to Streamlit Community Cloud.
- Add automated hyperparameter tuning via Optuna.
- Implement Evidently AI for data and concept drift monitoring.
- Set up automated CI/CD retraining triggers on data drift alerts.

---

## Author
**NNV Manikanta**  
* GitHub: [@NNV-MANIKANTA](https://github.com/NNV-MANIKANTA)  
* Project Repository: [customer-churn-mlops](https://github.com/NNV-MANIKANTA/customer-churn-mlops)

---

## License
This project is licensed under the MIT License for educational and portfolio purposes.
