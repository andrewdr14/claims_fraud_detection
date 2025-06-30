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

st.markdown("--- ")
st.header("Key Data Visualizations")
st.markdown("Visual insights into the dataset's characteristics and relationships.")

# 1. Target variable distribution (Fraud/Not Fraud)
st.markdown("#### Fraud Reported Distribution")
fig1, ax1 = plt.subplots(figsize=(6,4))
df['fraud_reported'].value_counts().plot(kind='bar', color=['#4caf50', '#f44336'], ax=ax1)
ax1.set_title('Fraud Reported Distribution')
ax1.set_xlabel('Fraud Reported')
ax1.set_ylabel('Count')
# Ensure labels are correctly mapped if fraud_reported is encoded
if 'fraud_reported' in category_mappings:
    ax1.set_xticks([0,1])
    ax1.set_xticklabels([category_mappings['fraud_reported'][i] for i in sorted(category_mappings['fraud_reported'])])
st.pyplot(fig1)

# 2. Fraud rate by categorical features
st.markdown("#### Fraud Rate by Key Categorical Features")
cat_features_to_plot = ['incident_severity', 'incident_type', 'collision_type', 'insured_sex', 'policy_state', 'auto_make']

for col in cat_features_to_plot:
    if col in df.columns and col in category_mappings:
        st.markdown(f"##### {col.replace("_", " ").title()}")
        fig, ax = plt.subplots(figsize=(8,4))
        fraud_rate = df.groupby(col)['fraud_reported'].mean()
        
        # Map codes back to categories for labeling
        labels = [category_mappings[col][i] for i in fraud_rate.index]
        sns.barplot(x=labels, y=fraud_rate.values, palette='viridis', ax=ax)
        
        ax.set_title(f'Fraud Rate by {col.replace("_", " ").title()}')
        ax.set_ylabel('Fraud Rate')
        ax.set_xlabel(col.replace("_", " ").title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# 3. Numerical features: Claim amounts
st.markdown("#### Distribution of Claim Amounts")
num_features_to_plot = ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']

for col in num_features_to_plot:
    if col in df.columns:
        st.markdown(f"##### {col.replace("_", " ").title()}")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col], bins=30, kde=True, color='steelblue', ax=ax)
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)

# 4. Boxplot: Claim amounts by Fraud Reported
st.markdown("#### Claim Amounts by Fraud Reported")
for col in num_features_to_plot:
    if col in df.columns and 'fraud_reported' in category_mappings:
        st.markdown(f"##### {col.replace("_", " ").title()} by Fraud Reported")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x=df['fraud_reported'].map(category_mappings['fraud_reported']), y=df[col], ax=ax)
        ax.set_title(f'{col.replace("_", " ").title()} by Fraud Reported')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel(col.replace("_", " ").title())
        plt.tight_layout()
        st.pyplot(fig)

# 5. Heatmap: Correlation matrix for numeric variables
st.markdown("#### Correlation Matrix of Numerical Features")
fig_corr, ax_corr = plt.subplots(figsize=(10,8))
sns.heatmap(df[num_features_to_plot + ['fraud_reported']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title('Correlation Matrix')
plt.tight_layout()
st.pyplot(fig_corr)