import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import json
import matplotlib.pyplot as plt
from model.explain_model import explain_prediction
from model.fairness_utils import check_fairness

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.explain_model import explain_prediction

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

import streamlit as st
import pandas as pd

def display_fairness_report(fairness_report):
    """
    Display the fairness report in a user-friendly format with explanations and tables.
    
    Parameters:
    - fairness_report (dict): Output from check_fairness(model)
    """

    st.write("### 📊 Fairness Analysis Report")
    st.write("This report analyzes how fairly the model performs across different patient attributes.")
    st.write("The following terms are used:")
    st.write("- **Accuracy by Group**: This is the model's prediction accuracy for each subgroup (e.g., Male, Female, etc.).")
    st.write("- **Accuracy Difference**: This is the difference in accuracy between the groups. A value closer to 0 indicates similar performance for both groups.")

    for attribute, data in fairness_report.items():
        st.markdown(f"#### 🧩 Fairness by {attribute}")
        
        # Explain the attribute
        if attribute == "Sex":
            st.markdown("**Sex** indicates whether the patient is male or female.")
        elif attribute == "Injury":
            st.markdown("**Injury** indicates whether the patient had an injury (1 = Yes, 0 = No).")
        elif attribute == "Pain":
            st.markdown("**Pain** indicates whether the patient reported pain (1 = Yes, 0 = No).")
        elif attribute == "Mental":
            st.markdown("**Mental status** represents the patient’s consciousness level on arrival (Alert, Verbal, Pain, Unresponsive).")
        
        # Display accuracy table
        group_accuracy = pd.DataFrame.from_dict(data["accuracy_by_group"], orient="index", columns=["Accuracy"])
        group_accuracy.index.name = attribute
        st.table(group_accuracy)

        # Explain the fairness
        diff = data["difference"]
        if diff == 0:
            st.success("✅ The model performs equally across all groups for this attribute.")
        elif diff < 0.01:
            st.info(f"ℹ️ Minor difference in performance between groups: **{diff:.4f}**")
        else:
            st.warning(f"⚠️ Noticeable difference in performance between groups: **{diff:.4f}**. Consider investigating this bias.")
        
        st.markdown("---")


import streamlit as st
import pandas as pd
import numpy as np

def display_shap_explanation(shap_vals, patient_id=0):
    st.markdown(f"## 🔍 SHAP Explanation for Patient {patient_id}")
    st.markdown("""
        SHAP (SHapley Additive exPlanations) provides insights into how each feature influenced the model’s decision.
        Positive values increase the risk, negative values reduce it.
    """)

    shap_values_all = shap_vals.get("shap_values_class_1", [])
    feature_names = shap_vals.get("feature_names", [])

    # Safety checks
    if not shap_values_all or not feature_names:
        st.error("Missing SHAP values or feature names.")
        return

    if patient_id >= len(shap_values_all):
        st.error("Invalid patient ID selected.")
        return

    # Get values for this patient
    shap_values = shap_values_all[patient_id]

    # Flatten SHAP values if 2D per feature
    if isinstance(shap_values[0], (list, np.ndarray)):
        shap_values = [sum(vals) for vals in shap_values]  # Sum or mean can be used depending on interpretation

    if len(shap_values) != len(feature_names):
        st.error("Mismatch between SHAP values and feature names.")
        return

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": np.round(shap_values, 4)
    })

    df["Impact Direction"] = df["SHAP Value"].apply(
        lambda x: "↑ Increased Risk" if x > 0 else "↓ Decreased Risk" if x < 0 else "No Effect"
    )

    df_sorted = df.reindex(df["SHAP Value"].abs().sort_values(ascending=False).index)

    st.write("### 🧠 Feature Contributions")
    st.table(df_sorted)

    st.write("### 📘 Summary of Top Influences")
    for _, row in df_sorted.head(3).iterrows():
        st.markdown(f"- **{row['Feature']}**: {row['Impact Direction']} (SHAP: {row['SHAP Value']})")

    st.info("These values show how much each feature influenced the model’s decision for this specific patient.")



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
        shap_vals = explain_prediction(model, df1)  # df1 is 1-row DataFrame for Patient 1
        top_feat = shap_vals[0]["top_contributing_feature"]
        top_val = shap_vals[0]["top_contributing_shap_value"]

        st.write(f"**Top contributing factor for Patient 1**: {top_feat} (SHAP value: {top_val})")
        st.write("This feature had the most significant impact on the model's prediction for this patient.")


    with st.expander("🔍 Reasoning for Patient 2"):
        shap_vals = explain_prediction(model, df1)  # df1 is 1-row DataFrame for Patient 1
        top_feat = shap_vals[0]["top_contributing_feature"]
        top_val = shap_vals[0]["top_contributing_shap_value"]

        st.write(f"**Top contributing factor for Patient 2**: {top_feat} (SHAP value: {top_val})")
        st.write("This feature had the most significant impact on the model's prediction for this patient.")

    with st.expander("📈 Check for Fairness"):
        fairness_report = check_fairness(model)
        display_fairness_report(fairness_report)
