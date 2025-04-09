def explain_prediction(model, input_df):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return f"Top contributing feature: {input_df.columns[abs(shap_values[1][0]).argmax()]}"
