# Customer Churn Prediction — MLOps End-to-End Project

## 🎯 Business Problem
Predict which telecom customers are likely to churn (cancel their subscription) based on their usage patterns and account details.

## 🏗️ Architecture
- **ML Models:** Logistic Regression, Decision Tree, Random Forest (Scikit-learn)
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Backend API:** FastAPI (Port 8000)
- **Frontend:** Streamlit (Port 8501)
- **Containerization:** Docker + Docker Compose
- **CI/CD:** GitHub Actions
- **Deployment:** AWS EC2 (Ubuntu 24.04)

## 🚀 Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python -m src.data.preprocess

# 4. Train models
python -m src.models.train

# 5. Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# 6. Start FastAPI
uvicorn api.main:app --reload --port 8000

# 7. Start Streamlit
streamlit run frontend/app.py

# 8. Run tests
pytest -v

# 9. Docker
docker compose up --build -d
```

## 📊 Dataset
- **Source:** IBM Telco Customer Churn
- **Records:** ~7,043
- **Features:** 20
- **Target:** Churn (Yes/No)

## 🌐 Access
- Streamlit UI: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

