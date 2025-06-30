import streamlit as st

st.set_page_config(
    page_title="Claims Fraud Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to the Claims Fraud Detection App!")

st.markdown("""
This interactive tool is designed to help you understand and compare different machine learning workflows for detecting fraudulent insurance claims.

### The Challenge of Insurance Fraud

Insurance fraud is a pervasive issue that costs the industry billions of dollars annually. It leads to higher premiums for honest policyholders and erodes trust in the insurance system. Traditional methods of fraud detection often rely on rule-based systems or manual investigations, which can be inefficient and prone to human error.

### The Machine Learning Solution

Machine learning offers a powerful approach to combat insurance fraud. By analyzing vast amounts of historical claims data, ML models can identify complex patterns and anomalies that indicate fraudulent activity. This allows for more accurate and efficient identification of suspicious claims, enabling insurers to take timely action.

### How This App Helps You Learn

This application provides a hands-on environment to explore the key stages of building a fraud detection system:

- **Data Exploration:** Understand the raw data, its characteristics, and how it's prepared for modeling.
- **Feature Engineering:** Learn how raw data is transformed into meaningful features that machine learning models can understand.
- **Model Comparison:** Experiment with different machine learning algorithms (like XGBoost and CatBoost) and feature selection techniques to see their impact on model performance.
- **Hyperparameter Tuning:** Discover how optimizing model parameters can significantly improve accuracy.

### Navigate the App

- **Compare Workflows:** Configure and run up to four different machine learning workflows side-by-side to compare their performance.

- **Data Explorer:** Explore the dataset used for this project, view key visualizations, and download the data for your own analysis.

- **Concepts:** Dive deeper into the machine learning models, feature selection methods, hyperparameter tuning, and imbalance handling techniques used in this application.

### How It Works: The ML Pipeline

Our fraud detection system follows a standard machine learning pipeline:

1.  **Data Collection & Preprocessing:** Raw insurance claims data is collected and then cleaned, handled for missing values, and transformed into a suitable format for machine learning. This includes encoding categorical variables and engineering new features.
2.  **Feature Selection:** Important features are identified and selected to improve model performance and reduce complexity. Techniques like Random Forest importance and L1 (Lasso) regularization are employed.
3.  **Model Training:** Machine learning models (XGBoost and CatBoost) are trained on the preprocessed and selected features. Options for hyperparameter tuning (Grid Search) and handling class imbalance (Oversampling, SMOTE, Class Weighting) are available.
4.  **Model Evaluation:** The trained models are rigorously evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrices to assess their effectiveness in identifying fraudulent claims.

### About This Project

This project was developed as an educational tool to demonstrate the practical application of machine learning in fraud detection. It aims to provide a transparent and interactive platform for users to experiment with different ML techniques and understand their impact on model outcomes. The code is open-source and designed to be easily understandable and extensible.

### Our Goal

Our primary goal is to provide an educational tool that demystifies machine learning in the context of fraud detection. We aim to offer a deeper understanding of how these powerful techniques work and how they can be applied to solve real-world problems.
""")