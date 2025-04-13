import shap
import pandas as pd
import joblib

# Load the feature columns (use the same one as in your main script)
feature_columns = joblib.load("model/feature_columns.pkl")

def explain_prediction(model, df):
    """
    Explain a model's prediction using SHAP.

    Parameters:
    - model: The trained machine learning model
    - df: The patient data for which the prediction is being explained (DataFrame)

    Returns:
    - A dictionary of SHAP values for explanation.
    """
    # Initialize SHAP explainer with the model
    explainer = shap.TreeExplainer(model)

    # Get SHAP values
    shap_values = explainer.shap_values(df)

    # Check the number of classes in the SHAP values
    if isinstance(shap_values, list):
        # Case 1: If SHAP values are returned as a list (for multi-class or binary classification)
        if len(shap_values) == 1:
            # Only one class (binary classification, might predict only 0 or 1)
            shap_vals = {
                "shap_values_class_1": shap_values[0].tolist(),  # SHAP values for class 0
                "shap_values_class_0": None,  # No class 1 in the prediction
                "feature_names": feature_columns
            }
        else:
            # Predicts both classes (0 and 1)
            shap_vals = {
                "shap_values_class_1": shap_values[1].tolist(),  # SHAP values for class 1
                "shap_values_class_0": shap_values[0].tolist(),  # SHAP values for class 0
                "feature_names": feature_columns
            }
    else:
        # Case 2: If SHAP values are returned as a single array (binary classification)
        shap_vals = {
            "shap_values_class_1": shap_values.tolist(),  # Only one class, so the same SHAP values
            "shap_values_class_0": None,  # No class 1 in the prediction
            "feature_names": feature_columns
        }

    return shap_vals

