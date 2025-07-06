import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

st.title("Data Explorer")

st.markdown("""
This section allows you to explore the cleaned insurance claims dataset used in this project. Understanding the data is crucial for building effective machine learning models.
""")

# Load raw dataset
raw_df = pd.read_csv("claims_fraud/insurance_claims.csv")

# Load cleaned dataset and mappings
df = pd.read_csv("claims_fraud/cleaned_insurance_claims.csv")

try:
    with open("claims_fraud/categorical_mappings.pkl", "rb") as f:
        category_mappings = pickle.load(f)
except FileNotFoundError:
    st.error("Categorical mappings file not found. Please run the `eda_insurance_claims_cleaned.ipynb` notebook first to generate it.")
    st.stop()


st.header("Raw Data")
st.markdown("The original, unprocessed insurance claims dataset.")
st.dataframe(raw_df)

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv_raw = convert_df(raw_df)

st.download_button(
    label="Download Raw Data as CSV",
    data=csv_raw,
    file_name='insurance_claims.csv',
    mime='text/csv',
)

st.markdown("--- ")

st.header("Cleaned Data")
st.markdown("The dataset after preprocessing, feature engineering, and encoding.")
st.dataframe(df)

csv_cleaned = convert_df(df)

st.download_button(
    label="Download Cleaned Data as CSV",
    data=csv_cleaned,
    file_name='cleaned_insurance_claims.csv',
    mime='text/csv',
)

st.markdown("--- ")
st.header("Exploratory Data Analysis Notebook")
st.markdown("""
To understand the data preprocessing, feature engineering, and initial statistical analysis, you can download the Jupyter Notebook used for Exploratory Data Analysis (EDA).
""")

# Read the notebook content
try:
    with open("claims_fraud/eda_insurance_claims_cleaned.ipynb", "rb") as f:
        notebook_content = f.read()
except FileNotFoundError:
    st.error("EDA Jupyter Notebook not found.")
    notebook_content = None

if notebook_content:
    st.download_button(
        label="Download EDA Jupyter Notebook",
        data=notebook_content,
        file_name="eda_insurance_claims_cleaned.ipynb",
        mime="application/x-ipynb+json",
    )