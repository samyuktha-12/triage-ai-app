import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from model.explain_model import explain_prediction
from model.fairness_utils import check_gender_fairness

# Load model and columns
model = joblib.load("model/triage_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

def preprocess_patient(patient):
    df = pd.DataFrame([patient])
    df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'})
    df['Injury'] = df['Injury'].map({1: 0, 2: 1})
    df['Pain'] = df['Pain'].map({0: 0, 1: 1})
    df['Mental'] = df['Mental'].map({1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'})
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

def display_shap_plot(df):
    explainer = shap.TreeExplainer(model)
    # Assuming shap_values is generated for class 0 or class 1
    shap_values = explainer.shap_values(df)

    # Check if only one class is predicted
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            # Only one class predicted, use shap_values[0] for the explanation
            st.pyplot(shap.summary_plot(shap_values[0], df, plot_type="bar", show=False))
        else:
            # Predicts both classes, so use shap_values[1] for class 1
            st.pyplot(shap.summary_plot(shap_values[1], df, plot_type="bar", show=False))
    else:
        # If shap_values is not a list, it's a binary classification with only one output
        st.pyplot(shap.summary_plot(shap_values, df, plot_type="bar", show=False))


st.title("🏥 Triage AI - Patient Prioritization System")

st.markdown("### 👤 Patient 1 Details")
with st.form("patient1_form"):
    p1 = {
        'KTAS duration_min': st.number_input("⏱️ Duration (min)", key='p1d'),
        'Length of stay_min': st.number_input("🏨 Length of Stay (min)", key='p1s'),
        'Sex': st.radio("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male", key='p1g'),
        'Injury': st.radio("Injury", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No", key='p1i'),
        'Pain': st.radio("Pain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='p1p'),
        'Mental': st.selectbox("Mental Status", [1, 2, 3, 4], format_func=lambda x: ["Alert", "Verbal", "Pain", "Unresponsive"][x-1], key='p1m')
    }
    st.form_submit_button("Submit", use_container_width=True)

st.markdown("### 👤 Patient 2 Details")
with st.form("patient2_form"):
    p2 = {
        'KTAS duration_min': st.number_input("⏱️ Duration (min)", key='p2d'),
        'Length of stay_min': st.number_input("🏨 Length of Stay (min)", key='p2s'),
        'Sex': st.radio("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male", key='p2g'),
        'Injury': st.radio("Injury", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No", key='p2i'),
        'Pain': st.radio("Pain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='p2p'),
        'Mental': st.selectbox("Mental Status", [1, 2, 3, 4], format_func=lambda x: ["Alert", "Verbal", "Pain", "Unresponsive"][x-1], key='p2m')
    }
    run = st.form_submit_button("Compare", use_container_width=True)

if 'run' in locals() and run:
    df1 = preprocess_patient(p1)
    df2 = preprocess_patient(p2)

    prob1 = model.predict_proba(df1)[0][1]
    prob2 = model.predict_proba(df2)[0][1]

    st.markdown(f"### 📊 Emergency Score: Patient 1 = `{prob1:.2f}`, Patient 2 = `{prob2:.2f}`")

    if prob1 > prob2:
        st.success("🚨 Patient 1 should be prioritized for the emergency room.")
    elif prob2 > prob1:
        st.success("🚨 Patient 2 should be prioritized for the emergency room.")
    else:
        st.warning("⚖️ Both patients have equal priority.")

    with st.expander("🔍 Reasoning for Patient 1"):
        shap_vals = explain_prediction(model, df1)
        st.json(shap_vals)
        display_shap_plot(df1)

    with st.expander("🔍 Reasoning for Patient 2"):
        shap_vals = explain_prediction(model, df2)
        st.json(shap_vals)
        display_shap_plot(df2)

    with st.expander("📈 Check for Gender Fairness"):
        fairness_report = check_gender_fairness(model)
        st.json(fairness_report)
