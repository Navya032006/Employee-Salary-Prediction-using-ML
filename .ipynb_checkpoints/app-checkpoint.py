import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🚀 Employee Salary Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple but stunning CSS that actually works in Streamlit
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

/* Main background */
.stApp {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Fix dropdown text visibility - MOST IMPORTANT */
.stSelectbox label {
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
}

.stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 3px solid #ffffff !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    color: #000000 !important;
    font-weight: bold !important;
    font-size: 16px !important;
}

/* Dropdown options */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 2px solid #667eea !important;
    border-radius: 10px !important;
}

div[data-baseweb="popover"] li {
    color: #000000 !important;
    font-weight: bold !important;
    padding: 10px 15px !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #667eea !important;
    color: white !important;
}

/* Number input styling */
.stNumberInput label {
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
}

.stNumberInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 3px solid #ffffff !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 10px !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #feca57) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 20px !important;
    padding: 15px 30px !important;
    border: none !important;
    border-radius: 25px !important;
    width: 100% !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
    transition: transform 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 30px rgba(0,0,0,0.4) !important;
}

/* Title styling */
.main-title {
    color: white;
    font-size: 4rem;
    font-weight: bold;
    text-align: center;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

.subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.4rem;
    text-align: center;
    margin-bottom: 40px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

/* Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    margin: 10px;
}

.metric-card h3 {
    color: white;
    font-size: 1.2rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.metric-card p {
    color: #feca57;
    font-size: 1.1rem;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

/* Form section */
.form-header {
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin: 30px 0;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
}

/* Prediction result */
.prediction-result {
    background: linear-gradient(45deg, #00b894, #00cec9);
    border: 3px solid white;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin: 20px 0;
    animation: slideIn 0.5s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(50px); }
    to { opacity: 1; transform: translateY(0); }
}

.prediction-result h2 {
    color: white;
    font-size: 2rem;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.salary-amount {
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin: 10px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.prediction-note {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1rem;
    margin-top: 20px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# Load model
model_data = joblib.load("salary_predictor.pkl")
model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Load supporting assets
eval_plot = Image.open("evaluation_plot.png")
with open("model_score.txt", "r") as f:
    r2_score = float(f.read())

# Header section
st.markdown('<div class="main-title">🚀 Employee Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Harness the power of machine learning to predict employee salaries with precision</div>', unsafe_allow_html=True)

# Info cards using Streamlit columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    <div class="metric-card">
        <h3>🧠 Algorithm</h3>
        <p>Linear Regression</p>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="metric-card">
        <h3>📊 Model Accuracy</h3>
        <p>R² Score: {r2_score:.4f}</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="metric-card">
        <h3>🎯 Prediction Type</h3>
        <p>Salary Estimation</p>
    </div>
    ''', unsafe_allow_html=True)

# Form section
st.markdown('<div class="form-header">📝 Enter Employee Details</div>', unsafe_allow_html=True)

# Input form
with st.form("salary_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("👤 Age", min_value=18, max_value=80, value=30)
        education_level = st.selectbox("🎓 Education Level", options=label_encoders["Education Level"].classes_)
        years_of_experience = st.number_input("💼 Years of Experience", min_value=0, max_value=40, value=5)
    
    with col2:
        gender = st.selectbox("⚧ Gender", options=label_encoders["Gender"].classes_)
        job_title = st.selectbox("💻 Job Title", options=label_encoders["Job Title"].classes_)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🔮 PREDICT MY SALARY")

# Prediction output
if submit_button:
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education_level],
        "Job Title": [job_title],
        "Years of Experience": [years_of_experience]
    })

    for col in ["Gender", "Education Level", "Job Title"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    predicted_salary = model.predict(input_scaled)[0]

    predicted_inr = predicted_salary * 83  # Approx conversion rate to INR
    
    st.markdown(f'''
    <div class="prediction-result">
        <h2>💰 Your Estimated Annual Salary</h2>
        <div class="salary-amount">USD ${predicted_salary:,.2f}</div>
        <div class="salary-amount">INR ₹{predicted_inr:,.0f}</div>
        <div class="prediction-note">
            🎯 This AI-powered prediction is based on your profile and our trained model<br>
            📈 Use this as a reference point for salary negotiations and career planning<br>
            ⭐ Results may vary based on location, company size, and market conditions
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Evaluation section
st.markdown('<div class="form-header">📈 Model Performance</div>', unsafe_allow_html=True)
st.image(eval_plot, caption="🔍 Model Evaluation: Actual vs Predicted Salary Analysis", use_container_width=True)