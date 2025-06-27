# Claims Fraud Detection

A machine learning project focused on detecting fraudulent insurance claims. This repository provides a comprehensive pipeline from data exploration and feature engineering to model training, evaluation, and an interactive Streamlit dashboard for comparing different ML workflows.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Exploratory Data Analysis (Jupyter Notebook)](#exploratory-data-analysis-jupyter-notebook)
  - [Interactive Application (Streamlit)](#interactive-application-streamlit)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Improvements](#improvements)
- [References](#references)

---

## Project Overview

Insurance fraud poses a significant challenge to the industry. This project aims to leverage machine learning to accurately identify potentially fraudulent claims. It provides a practical demonstration of building an end-to-end fraud detection system, emphasizing advanced modeling techniques and careful feature selection.

Key aspects include:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) to understand data characteristics
- Feature selection using various methods (Random Forest, L1 Regularization)
- Model training with robust algorithms like XGBoost and CatBoost
- Hyperparameter tuning with GridSearchCV for optimal performance
- An interactive Streamlit application for side-by-side comparison of different ML workflows.

---

## Features

- **Interactive Streamlit App**: Compare multiple machine learning workflows, including different models, feature selection methods, hyperparameter tuning strategies, and imbalance handling techniques.
- **Comprehensive ML Pipeline**: Demonstrates data splitting, feature selection, model training, and evaluation.
- **Imbalance Handling**: Includes techniques like Random Oversampling, SMOTE, and Class Weighting to address imbalanced datasets common in fraud detection.
- **Detailed EDA**: A Jupyter Notebook provides in-depth data exploration, cleaning, and feature engineering steps.
- **Model Comparison**: Easily visualize and compare performance metrics (accuracy, classification reports, confusion matrices) of different models.
- **Configurable Randomness**: Allows users to set a `random_state` for reproducibility or to observe variations in model training.

---

## Project Structure

```
claims_fraud_detection/
│
├── claims_fraud/
│   ├── __init__.py
│   ├── cleaned_insurance_claims.csv
│   ├── eda_insurance_claims_cleaned.ipynb
│   ├── insurance_claims.csv
│   ├── interactive_app.py (Note: Main app is Home.py, this is a placeholder/legacy reference)
│   └── model.py
│
├── pages/
│   ├── 1_Compare_Workflows.py
│   ├── 2_Data_Explorer.py
│   └── 3_Concepts.py
│
├── .gitignore
├── Home.py
├── README.md
├── requirements.txt
└── TODO
```

- **`claims_fraud/`**: Contains core ML scripts and data.
  - **`eda_insurance_claims_cleaned.ipynb`**: Jupyter Notebook for EDA, data cleaning, and feature engineering.
  - **`model.py`**: Core functions for data splitting, feature selection, model training (XGBoost, CatBoost), and evaluation.
  - **`insurance_claims.csv`**: The raw dataset.
  - **`cleaned_insurance_claims.csv`**: The cleaned and preprocessed dataset.
- **`pages/`**: Streamlit application pages.
  - **`1_Compare_Workflows.py`**: The main interactive page for comparing ML workflows.
  - **`2_Data_Explorer.py`**: Page for exploring the raw and cleaned datasets with visualizations.
  - **`3_Concepts.py`**: Explanations of machine learning concepts used in the project.
- **`Home.py`**: The main entry point for the Streamlit multi-page application.
- **`requirements.txt`**: Lists all Python dependencies.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andrewdr14/claims_fraud_detection.git
    cd claims_fraud_detection
    ```

2.  **Install dependencies** (Python 3.8+ recommended):
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

There are two primary ways to interact with this project:

### 1. Exploratory Data Analysis (Jupyter Notebook)

To delve into the data preprocessing, feature engineering, and initial statistical analysis, run the Jupyter Notebook:

1.  **Navigate to the `claims_fraud` directory:**
    ```bash
    cd claims_fraud
    ```

2.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

3.  In the Jupyter interface, open `eda_insurance_claims_cleaned.ipynb`.

### 2. Interactive Application (Streamlit)

The main interactive component is the Streamlit application, which allows you to compare different machine learning models and feature selection techniques in real-time.

1.  **Ensure you are in the root directory of the project (`claims_fraud_detection`):**
    ```bash
    cd claims_fraud_detection
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run Home.py
    ```

This command will open the interactive dashboard in your web browser, typically at `http://localhost:8501`.

---

## Machine Learning Workflow

This project implements a robust ML workflow:

-   **Data Splitting**: Divides data into training and testing sets.
-   **Feature Selection**: Utilizes Random Forest importance and L1 (Lasso) regularization to identify and select the most impactful features.
-   **Imbalance Handling**: Addresses class imbalance using techniques like Random Oversampling, SMOTE, or Class Weighting to improve minority class prediction.
-   **Model Training**: Employs powerful gradient boosting algorithms:
    -   **XGBoost (Extreme Gradient Boosting)**: Known for its performance and speed.
    -   **CatBoost**: Excels at handling categorical features natively.
-   **Hyperparameter Tuning**: Uses GridSearchCV to find optimal model parameters.
-   **Evaluation**: Models are assessed using standard metrics such as accuracy, classification reports, and confusion matrices.

---

## Improvements

-   **Configurable Random State**: The `random_state` for data splitting, feature selection, and model training is now configurable within the Streamlit app, allowing for more flexible experimentation and verification of model behavior.

---

## References

-   Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
-   Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
-   scikit-learn documentation: https://scikit-learn.org/
-   XGBoost documentation: https://xgboost.readthedocs.io/
-   CatBoost documentation: https://catboost.ai/