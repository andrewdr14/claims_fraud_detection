import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("Insurance Fraud Model Results")

# --- Feature Importance ---
st.header("Random Forest Feature Importances")
if os.path.exists("rf_selected_feature_importances.png"):
    st.image("rf_selected_feature_importances.png", caption="Selected Feature Importances (RandomForest)", use_column_width=True)
else:
    st.warning("Feature importances plot not found.")

# --- Omitted Features ---
st.header("Omitted Features")
if os.path.exists("omitted_features.txt"):
    with open("omitted_features.txt", "r") as f:
        omitted_features = f.read().strip()
    if omitted_features:
        st.code(omitted_features)
    else:
        st.write("No omitted features.")
else:
    st.warning("Omitted features file not found.")

# --- Pie Charts ---
st.header("Fraud Distribution Pie Charts")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("fraud_pie_incident_severity.png"):
        st.image("fraud_pie_incident_severity.png", caption="Fraud by Incident Severity", use_column_width=True)
    else:
        st.warning("Incident severity pie chart not found.")
with col2:
    if os.path.exists("fraud_pie_insured_hobbies.png"):
        st.image("fraud_pie_insured_hobbies.png", caption="Fraud by Insured Hobbies", use_column_width=True)
    else:
        st.warning("Insured hobbies pie chart not found.")

# --- Confusion Matrices ---
st.header("Confusion Matrices")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("confusion_matrix_default.png"):
        st.subheader("XGBoost (Default)")
        st.image("confusion_matrix_default.png", use_column_width=True)
    else:
        st.warning("Default confusion matrix not found.")
with col2:
    if os.path.exists("confusion_matrix_gridsearch.png"):
        st.subheader("XGBoost (Grid Search)")
        st.image("confusion_matrix_gridsearch.png", use_column_width=True)
    else:
        st.warning("Grid search confusion matrix not found.")

# --- Classification Reports ---
st.header("Classification Reports")
col1, col2 = st.columns(2)
with col1:
    st.subheader("XGBoost (Default)")
    if os.path.exists("classification_report_default.txt"):
        with open("classification_report_default.txt", "r") as f:
            st.code(f.read())
    else:
        st.warning("Default classification report not found.")
with col2:
    st.subheader("XGBoost (Grid Search)")
    if os.path.exists("classification_report_gridsearch.txt"):
        with open("classification_report_gridsearch.txt", "r") as f:
            st.code(f.read())
    else:
        st.warning("Grid Search classification report not found.")

# --- Best Grid Search Params ---
st.header("Best Grid Search XGBoost Parameters")
if os.path.exists("best_xgboost_params.txt"):
    with open("best_xgboost_params.txt", "r") as f:
        st.code(f.read())
else:
    st.warning("Best XGBoost params file not found.")

st.success("All results loaded! Scroll and expand panels as needed.")