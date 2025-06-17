import os
import joblib
import pandas as pd
from flask import Flask, render_template, send_file
from pymongo import MongoClient
from dotenv import load_dotenv
from . import model  # Import trained models (change to import model if running as a script!)
from sklearn.metrics import classification_report
from typing import Dict, Any

# Set template_folder to find templates in the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

# Ensure models are initialized before accessing features
model.initialize_models()
trained_features = model.trained_features

# Load pre-trained models
rf_model = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost.pkl")

@app.route("/")
def evaluation() -> str:
    """
    Run fraud model evaluation and display results.

    Returns:
        str: Rendered HTML template with evaluation metrics and summary statistics.
    """
    df = model.fetch_data()  # Load claims data

    df["fraud_reported"] = df["fraud_reported"].astype(str).str.strip().str.capitalize()
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)

    y_true = df["fraud_label"]
    y_pred_rf = rf_model.predict(df[trained_features])
    y_pred_xgb = xgb_model.predict(df[trained_features])

    # Generate classification reports dynamically
    rf_results = classification_report(y_true, y_pred_rf, output_dict=True)
    xgb_results = classification_report(y_true, y_pred_xgb, output_dict=True)

    # Generate summary statistics
    policy_holder_count = len(df)
    fraud_cases = (df["fraud_reported"] == "Yes").sum()
    non_fraud_cases = (df["fraud_reported"] == "No").sum()

    # Compute descriptive statistics for numeric columns (excluding specific ones)
    excluded_columns = ["umbrella_limit", "incident_hour_of_the_day", "number_of_vehicles", "fraud_label"]
    numeric_cols = [col for col in trained_features if col not in excluded_columns]

    summary_stats: Dict[str, Any] = {
        "Total Policy Holders": policy_holder_count,
        "Fraud Reported": fraud_cases,
        "No Fraud Reported": non_fraud_cases
    }

    # Add detailed statistics (Max, Min, Mean, Median, Std Dev)
    for col in numeric_cols:
        summary_stats[f"{col} - Min"] = df[col].min()
        summary_stats[f"{col} - Max"] = df[col].max()
        summary_stats[f"{col} - Mean"] = round(df[col].mean(), 2)
        summary_stats[f"{col} - Median"] = df[col].median()
        summary_stats[f"{col} - Std Dev"] = round(df[col].std(), 2)

    return render_template(
        "results.html",
        rf_results=rf_results,
        xgb_results=xgb_results,
        summary_stats=summary_stats
    )

@app.route("/download-data")
def download_data() -> Any:
    """
    Allow users to download the dataset.

    Returns:
        Response: Sends the CSV dataset file as an attachment.
    """
    # Ensure the CSV is served from the correct directory
    csv_path = os.path.join(os.path.dirname(__file__), "motor_insurance_claims.csv")
    if not os.path.exists(csv_path):
        # Return a 404 with a simple error message if the file doesn't exist
        return "Dataset not found. Please generate data first.", 404
    return send_file(csv_path, as_attachment=True)

if __name__ == "__main__":
    # Dynamically assign Render's port for deployment
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)