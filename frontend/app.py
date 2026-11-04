"""
Customer Churn Prediction System - Streamlit Frontend
Integrated with FastAPI Backend via REST API.
"""

import os
import requests
import streamlit as st

# Configure API Base URL from environment or default to local FastAPI
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
MODEL_INFO_ENDPOINT = f"{API_BASE_URL}/model-info"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def check_api_health():
    """Checks whether the FastAPI backend is online and responding."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("status", "healthy"), data.get("model_loaded", False)
        return False, f"Status code {response.status_code}", False
    except requests.exceptions.ConnectionError:
        return False, "Connection refused", False
    except requests.exceptions.Timeout:
        return False, "Request timed out", False
    except Exception as e:
        return False, str(e), False


def fetch_model_info():
    """Fetches model metrics and metadata from the FastAPI backend."""
    try:
        response = requests.get(MODEL_INFO_ENDPOINT, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def send_prediction_request(payload: dict):
    """
    Sends customer data payload to FastAPI /predict endpoint.
    Returns: (success: bool, data: dict, error_message: str)
    """
    try:
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=5)
        if response.status_code == 200:
            return True, response.json(), ""
        elif response.status_code == 422:
            return False, {}, f"Validation error (422): Input does not match backend schema. Details: {response.text}"
        else:
            return False, {}, f"API returned error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return (
            False,
            {},
            f"Cannot connect to FastAPI at {API_BASE_URL}. "
            "Please ensure the backend is running (e.g., 'uvicorn api.main:app --port 8000')."
        )
    except requests.exceptions.Timeout:
        return False, {}, f"Request to {PREDICT_ENDPOINT} timed out after 5 seconds."
    except Exception as e:
        return False, {}, f"An unexpected error occurred: {str(e)}"


# -------------------------------------------------------------------------
# Sidebar: Backend Connection Status & Profiles
# -------------------------------------------------------------------------
st.sidebar.title("⚙️ System Status")

is_online, health_msg, is_loaded = check_api_health()
if is_online and is_loaded:
    st.sidebar.success(f"🟢 Backend Online: {API_BASE_URL}")
    st.sidebar.caption("Pipeline loaded & ready for inference")
elif is_online and not is_loaded:
    st.sidebar.warning("🟡 Backend Online (Model not loaded)")
else:
    st.sidebar.error(f"🔴 Backend Offline: {API_BASE_URL}")
    st.sidebar.caption(f"Reason: {health_msg}")

st.sidebar.divider()

# Demo Profile Quick-loader
st.sidebar.subheader("📋 Load Sample Profile")
preset = st.sidebar.selectbox(
    "Choose a preset to autofill the form:",
    ["Default Form", "High Risk Customer (Likely to Churn)", "Loyal Customer (Low Risk)"]
)

model_info = fetch_model_info()
if model_info:
    st.sidebar.divider()
    st.sidebar.subheader("🧠 Model Metadata")
    st.sidebar.write(f"**Best Model:** `{model_info.get('best_model')}`")
    metrics = model_info.get("metrics", {})
    if metrics:
        st.sidebar.write(f"• Accuracy: `{metrics.get('accuracy', 0):.2%}`")
        st.sidebar.write(f"• F1-Score: `{metrics.get('f1_score', 0):.4f}`")
        st.sidebar.write(f"• ROC-AUC: `{metrics.get('roc_auc', 0):.4f}`")

# -------------------------------------------------------------------------
# Main Page Header
# -------------------------------------------------------------------------
st.markdown('<div class="main-header">Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered enterprise risk assessment dashboard. '
    'Predict customer churn probability and generate proactive retention actions via our production MLOps pipeline.</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# Preset values definition
# -------------------------------------------------------------------------
if preset == "High Risk Customer (Likely to Churn)":
    p_gender = "Female"
    p_senior = 0
    p_partner = "No"
    p_dependents = "No"
    p_tenure = 1
    p_phone = "Yes"
    p_lines = "No"
    p_internet = "Fiber optic"
    p_security = "No"
    p_backup = "No"
    p_protection = "No"
    p_support = "No"
    p_tv = "Yes"
    p_movies = "Yes"
    p_contract = "Month-to-month"
    p_paperless = "Yes"
    p_payment = "Electronic check"
    p_monthly = 85.50
    p_total = 85.50
elif preset == "Loyal Customer (Low Risk)":
    p_gender = "Male"
    p_senior = 0
    p_partner = "Yes"
    p_dependents = "Yes"
    p_tenure = 60
    p_phone = "Yes"
    p_lines = "Yes"
    p_internet = "DSL"
    p_security = "Yes"
    p_backup = "Yes"
    p_protection = "Yes"
    p_support = "Yes"
    p_tv = "Yes"
    p_movies = "Yes"
    p_contract = "Two year"
    p_paperless = "No"
    p_payment = "Credit card (automatic)"
    p_monthly = 65.00
    p_total = 3900.00
else:
    # Default Form
    p_gender = "Male"
    p_senior = 0
    p_partner = "No"
    p_dependents = "No"
    p_tenure = 12
    p_phone = "Yes"
    p_lines = "No"
    p_internet = "Fiber optic"
    p_security = "No"
    p_backup = "No"
    p_protection = "No"
    p_support = "No"
    p_tv = "No"
    p_movies = "No"
    p_contract = "Month-to-month"
    p_paperless = "Yes"
    p_payment = "Electronic check"
    p_monthly = 70.00
    p_total = 840.00

# -------------------------------------------------------------------------
# Customer Input Form
# -------------------------------------------------------------------------
with st.form("churn_prediction_form"):
    st.subheader("1. Customer Profile & Demographics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if p_gender == "Male" else 1)
    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            index=p_senior
        )
    with col3:
        partner = st.selectbox("Has Partner?", ["No", "Yes"], index=0 if p_partner == "No" else 1)
    with col4:
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0 if p_dependents == "No" else 1)

    st.subheader("2. Telephony & Core Internet Services")
    col5, col6, col7 = st.columns(3)
    with col5:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if p_phone == "Yes" else 1)
    with col6:
        lines_opts = ["No", "Yes", "No phone service"]
        multiple_lines = st.selectbox("Multiple Lines", lines_opts, index=lines_opts.index(p_lines))
    with col7:
        internet_opts = ["Fiber optic", "DSL", "No"]
        internet_service = st.selectbox("Internet Service", internet_opts, index=internet_opts.index(p_internet))

    st.subheader("3. Add-on & Security Services")
    col8, col9, col10 = st.columns(3)
    addon_opts = ["No", "Yes", "No internet service"]
    with col8:
        online_security = st.selectbox("Online Security", addon_opts, index=addon_opts.index(p_security))
        online_backup = st.selectbox("Online Backup", addon_opts, index=addon_opts.index(p_backup))
    with col9:
        device_protection = st.selectbox("Device Protection", addon_opts, index=addon_opts.index(p_protection))
        tech_support = st.selectbox("Tech Support", addon_opts, index=addon_opts.index(p_support))
    with col10:
        streaming_tv = st.selectbox("Streaming TV", addon_opts, index=addon_opts.index(p_tv))
        streaming_movies = st.selectbox("Streaming Movies", addon_opts, index=addon_opts.index(p_movies))

    st.subheader("4. Contract, Billing & Charges")
    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        contract_opts = ["Month-to-month", "One year", "Two year"]
        contract = st.selectbox("Contract Type", contract_opts, index=contract_opts.index(p_contract))
    with col12:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if p_paperless == "Yes" else 1)
    with col13:
        payment_opts = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
        payment_method = st.selectbox("Payment Method", payment_opts, index=payment_opts.index(p_payment))
    with col14:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=p_tenure, step=1)
    with col15:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=float(p_monthly), step=0.5)

    col_tot1, col_tot2 = st.columns([1, 3])
    with col_tot1:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(p_total), step=1.0)
    with col_tot2:
        st.caption("ℹ️ Tip: Total charges typically approximate `Tenure × Monthly Charges` for consistent accounts.")

    st.write("")
    submit_button = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

# -------------------------------------------------------------------------
# Handle Prediction Submission
# -------------------------------------------------------------------------
if submit_button:
    # Build payload with exact types matching FastAPI CustomerInput schema
    payload = {
        "gender": str(gender),
        "SeniorCitizen": int(senior_citizen),
        "Partner": str(partner),
        "Dependents": str(dependents),
        "tenure": int(tenure),
        "PhoneService": str(phone_service),
        "MultipleLines": str(multiple_lines),
        "InternetService": str(internet_service),
        "OnlineSecurity": str(online_security),
        "OnlineBackup": str(online_backup),
        "DeviceProtection": str(device_protection),
        "TechSupport": str(tech_support),
        "StreamingTV": str(streaming_tv),
        "StreamingMovies": str(streaming_movies),
        "Contract": str(contract),
        "PaperlessBilling": str(paperless_billing),
        "PaymentMethod": str(payment_method),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    with st.spinner("Contacting FastAPI inference engine..."):
        success, result, error_msg = send_prediction_request(payload)

    if not success:
        st.error(f"❌ Prediction Failed: {error_msg}")
    else:
        st.success("✅ Prediction Successfully Generated from FastAPI Backend")
        st.divider()

        prediction = result.get("churn_prediction")
        churn_label = result.get("churn_label", "Unknown")
        probability = result.get("churn_probability", 0.0)
        risk_level = result.get("risk_level", "Medium")
        recommendation = result.get("retention_recommendation", "N/A")

        # Visual Dashboard Cards
        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            st.markdown("### Prediction Outcome")
            if churn_label == "Yes":
                st.error("⚠️ **CUSTOMER WILL CHURN**")
            else:
                st.success("🛡️ **CUSTOMER WILL STAY (RETAINED)**")

        with res_col2:
            st.markdown("### Churn Probability")
            pct_val = probability * 100
            st.metric(label="Calculated Probability", value=f"{pct_val:.2f}%")
            st.progress(min(max(float(probability), 0.0), 1.0))

        with res_col3:
            st.markdown("### Risk Level")
            if risk_level == "High":
                st.markdown("<h3 style='color:#dc2626;'>🚨 HIGH RISK</h3>", unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.markdown("<h3 style='color:#d97706;'>⚠️ MEDIUM RISK</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:#16a34a;'>✅ LOW RISK</h3>", unsafe_allow_html=True)

        st.write("")
        st.info(f"💡 **Retention Recommendation:** {recommendation}")

        with st.expander("🔍 View Raw JSON Response from FastAPI"):
            st.json(result)
