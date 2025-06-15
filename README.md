# Motor Insurance Fraud Detection System

This project analyzes insurance claims for fraud using machine learning. It demonstrates an end-to-end pipeline with three main components:
- **`data_generator.py`**: Generates or processes a synthetic/real dataset of motor insurance claims.
- **`model.py`**: Connects to MongoDB, preprocesses data, trains machine learning models (Random Forest & XGBoost), and saves the trained models.
- **`app.py`**: Provides a Flask-based web interface that uses the trained models to evaluate claims, display performance metrics, and allow users to download the dataset.

The project integrates with **MongoDB** to store and retrieve claims data, ensuring a robust and dynamic data pipeline. Using MongoDB is not imperative to this project, but I just wanted to integrate it into my project.

---

## Project Overview

This project demonstrates how to:
- **Generate or Ingest Data**: Create synthetic data or process real-world claims data using `data_generator.py`.
- **Train Machine Learning Models**: Preprocess the data and train two popular models:
  - **Random Forest**: An ensemble learning method effective for structured data.
  - **XGBoost**: A gradient boosting algorithm optimized for performance.
- **Evaluate & Visualize Results**: Use a Flask web interface to compare models, view summary statistics (such as policy holders count, descriptive statistics, fraud vs. non-fraud counts), and download the processed dataset.

---

## Set-up and Execution

### 1. Clone the Repository and Install Dependencies

Clone the repository:

On a CLI: git clone https://github.com/<your-username>/claims-fraud-detection.git

On a CLI: cd claims-fraud-detection


Install the required packages:

On a CLI: pip install -r requirements.txt


### 2. Configure the MongoDB Connection

This project uses MongoDB to store and retrieve data.

Create a `.env` file in the project root (do not commit this file) and add your MongoDB URI:

MONGO_URI=<your-mongodb-connection-string>


### 3. Generate the Data

Run the data generator script to create or update the dataset (e.g., `motor_insurance_claims.csv`):

On a CLI: python data_generator.py


### 4. Train the Models

Train the machine learning models by running:

On a CLI: python model.py

This script will:
- Connect to MongoDB to fetch the claims data.
- Preprocess the data.
- Train the Random Forest and XGBoost models.
- Save the trained models (e.g., as `random_forest.pkl` and `xgboost.pkl`).

### 5. Launch the Web Application

Start the Flask web interface:

On a CLI: python app.py

Once running, your terminal will display a local URL. Open this URL in your browser to:
- View the fraud detection analysis.
- Compare model performance (Random Forest vs. XGBoost).
- Review statistical analyses (policy holder counts; min, max, median, mean, standard deviation for key features).
- Download the processed dataset.

---

## Fraud Detection Models

###  Random Forest
- **Type:** Ensemble learning method.
- **Strengths:** Robust and interpretable for structured data.
- **Evaluation Metrics:** Precision, Recall, and F1-Score.

###  XGBoost
- **Type:** Gradient boosting algorithm.
- **Strengths:** Optimized for large datasets and performance.
- **Evaluation Metrics:** Precision, Recall, and F1-Score (with fine-tuning for optimal log loss).

---

##  Statistical Analysis

The web interface provides:
- **Policy Holder Statistics:** Total policy holders, number of claims with fraud reported, and number without fraud.
- **Descriptive Statistics:** For each key feature you’ll see the minimum, maximum, median, mean, and standard deviation.
- **Model Evaluation:** Separate evaluation tables for Random Forest and XGBoost with metrics rounded to two decimal places.
- **Definitions:** Explanations for evaluation metrics (Precision, Recall, F1-Score) to help interpret the results.

---

## 📝 Repository Structure

```
claims-fraud-detection
│──  app.py              # Flask web application for evaluating and displaying results
│──  model.py            # Data preprocessing, MongoDB connection, and model training
│──  data_generator.py   # Script to generate or process synthetic/real datasets
│──  templates/          # HTML templates for the web interface (e.g., results.html)
│──  static/             # CSS and other static assets (e.g., styles.css)
│──  	requirements.txt    # Python dependencies for the project
│──  README.md           # This documentation file
│──  .env                # Environment variables (e.g., MONGO_URI; not tracked by Git)
```

---

## Future Enhancements

- **Feature Importance Visualization:** Use SHAP or similar methods to showcase which features drive model predictions.
- **Hyperparameter Tuning:** Implement grid or random search to optimize model performance.
- **Additional Evaluation Metrics:** Expand the evaluation to include ROC curves, calibration plots, and confusion matrices.
- **Enhanced Data Generation:** Improve data_generator.py to simulate more complex fraud scenarios.
- **Interactive Dashboards:** Develop interactive visualizations for deeper insights into model performance.

---

## Contributing

Contributions are welcome. To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Open a pull request detailing your contribution.


