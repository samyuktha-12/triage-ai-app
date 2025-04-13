import shap
import joblib
import numpy as np

feature_columns = joblib.load("model/feature_columns.pkl")

def explain_prediction(model, df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    class_1_shap = shap_values[1] if isinstance(shap_values, list) else shap_values

    results = []

    for i in range(df.shape[0]):
        patient_shap = class_1_shap[i]

        if isinstance(patient_shap[0], (list, np.ndarray)):
            patient_shap = np.array([val[1] for val in patient_shap]) 

        max_idx = int(np.argmax(np.abs(patient_shap)))
        results.append({
            "top_contributing_feature": feature_columns[max_idx],
            "top_contributing_shap_value": round(patient_shap[max_idx], 6)
        })

    return results
