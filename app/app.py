import streamlit as st
import pandas as pd
import joblib
from model.explain_model import explain_prediction
from model.fairness_utils import apply_fairness

st.title("🚑 AI-Powered Triage Simulator")

st.sidebar.header("Patient Information")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100)
arrival_mode = st.sidebar.selectbox("Arrival Mode", ["Walking", "Public Ambulance", "Private Vehicle"])
injury = st.sidebar.checkbox("Injury")
mental = st.sidebar.selectbox("Mental State", ["Alert", "Verbal", "Pain", "Unresponsive"])
pain = st.sidebar.checkbox("In Pain?")
sbp = st.sidebar.slider("Systolic BP", 80, 200)
dbp = st.sidebar.slider("Diastolic BP", 40, 140)
hr = st.sidebar.slider("Heart Rate", 40, 180)
rr = st.sidebar.slider("Respiratory Rate", 10, 40)
bt = st.sidebar.slider("Body Temp (°C)", 35.0, 42.0)

# Prepare input
input_data = pd.DataFrame([{
    'Sex_Male': 1 if sex == "Male" else 0,
    'Age': age,
    'Arrival_Mode_Public Ambulance': 1 if arrival_mode == "Public Ambulance" else 0,
    'Injury': int(injury),
    'Mental_Alert': 1 if mental == "Alert" else 0,
    'Pain': int(pain),
    'SBP': sbp,
    'DBP': dbp,
    'HR': hr,
    'RR': rr,
    'BT': bt
}])

# Load baseline model
model = joblib.load("model/triage_model.pkl")
baseline_pred = model.predict(input_data)[0]

# Show predictions
st.subheader("⚖️ AI Decision")
st.write(f"**Baseline Prediction:** {'Emergency' if baseline_pred else 'Non-Emergency'}")

# Add fairness toggle
if st.checkbox("Compare with Fairness Mitigation"):
    fair_pred = apply_fairness(...)  # Add actual inputs for fairness function
    st.write(f"**Fair AI Prediction:** {'Emergency' if fair_pred.labels[0][0] else 'Non-Emergency'}")

# Explanation
st.markdown("### 🔍 Model Explanation")
explanation = explain_prediction(model, input_data)
st.write(explanation)
