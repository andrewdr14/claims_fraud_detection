# Claims Fraud Detection

A machine learning project for detecting fraudulent insurance claims. This repository contains code and notebooks for data exploration, feature engineering, model training, evaluation, and dashboard visualizations. The workflow leverages ensemble learning techniques, hyperparameter optimization, and interpretable outputs to deliver a robust fraud detection pipeline.

---

## Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Theory and Modeling Workflow](#machine-learning-theory-and-modeling-workflow)
- [References](#references)
- [License](#license)

---

## Project Overview

Insurance fraud is a significant challenge in the industry, leading to substantial financial losses each year. This project aims to build a machine learning pipeline that accurately identifies potentially fraudulent claims, using advanced modeling techniques and careful feature selection.

Key steps include:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature selection using Random Forest
- Model training using XGBoost
- Hyperparameter tuning with GridSearchCV
- Final model evaluation and deployment

---

## Data Description

The dataset consists of anonymized insurance claim records, including:
- Categorical and numerical policyholder features
- Claim details (e.g. amount, type, timing)
- Labeled outcome indicating whether the claim was fraudulent

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/andrewdr14/claims_fraud_detection.git
   cd claims_fraud_detection
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. To use Jupyter notebooks:
   ```bash
   pip install jupyter
   ```

---

## Usage

- **Data exploration and cleaning:**
  - Run `eda_and_cleaning.ipynb` for an overview of data preprocessing, feature engineering, and statistical analysis.

- **Model training and evaluation:**
  - Execute `model.py` or the accompanying notebook to:
    - Select features with Random Forest
    - Train XGBoost models
    - Optimize hyperparameters with GridSearchCV
    - Evaluate model performance

- **Dashboard:**
  - Launch the dashboard app (if provided) to visualize predictions, feature importances, and other metrics.

---

## Machine Learning Theory and Modeling Workflow

### Model Theory

- **Random Forest (RF):**  
  Random Forest is an ensemble learning method that builds multiple decision trees and merges their results to improve accuracy and control overfitting. It handles non-linear relationships well and is robust to noise and outliers. In this project, Random Forest is used specifically for **feature selection**, ranking the importance of input variables and helping to reduce dimensionality before further modeling.

- **XGBoost (Extreme Gradient Boosting):**  
  XGBoost is an advanced and efficient implementation of gradient-boosted trees, designed for speed and performance. It builds trees sequentially, where each new tree corrects errors from the previous ones, and includes regularization to prevent overfitting. XGBoost is particularly effective for structured, tabular data and is widely used in machine learning competitions.

- **GridSearchCV:**  
  GridSearchCV is a hyperparameter optimization technique from scikit-learn. It exhaustively searches over specified parameter values for an estimator, using cross-validation to evaluate model performance for each parameter combination. This process ensures that the chosen model has the most effective configuration for the task.

### Modeling Workflow in This Project

1. **Feature Selection with Random Forest:**  
   After data preprocessing and cleaning, a Random Forest classifier is trained to evaluate the importance of each feature. The top-ranked features are selected for downstream modeling, which helps improve model accuracy and reduces computational cost.

2. **Initial XGBoost Modeling:**  
   XGBoost is trained using the selected features from the Random Forest step. This creates a strong baseline model and helps highlight which hyperparameters may benefit from further tuning.

3. **Hyperparameter Tuning with GridSearchCV:**  
   The XGBoost model is wrapped with GridSearchCV to systematically explore combinations of hyperparameters (such as learning rate, tree depth, etc.). This step identifies the most effective configuration for the XGBoost algorithm on the fraud detection dataset.

4. **Final Modeling with XGBoost:**  
   Using the best hyperparameters found by GridSearchCV, XGBoost is retrained on the data to produce the final, optimized model. The performance of this model is then evaluated and visualized as part of the project output.

---

## References

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/

---
