"""
app.py

This is the main Flask application for the insurance fraud project.
- At startup, it uploads the generated CSV and model results to MongoDB (if MONGO_URI is set).
- The website displays results from the local files, NOT from MongoDB.
- MongoDB is used as an archive/backup only.

Usage:
    flask run
    or (recommended for production):
    gunicorn --bind 0.0.0.0:8080 claims_fraud.app:app
"""

import os
import pandas as pd
import pickle
from flask import Flask, render_template
from pymongo import MongoClient

app = Flask(__name__)

def upload_to_mongodb(csv_path, results_path, mongo_uri):
    """
    Uploads the CSV data and model results to MongoDB.

    Args:
        csv_path (str): Path to the generated CSV file.
        results_path (str): Path to the pickled model results.
        mongo_uri (str): MongoDB connection URI.
    """
    client = MongoClient(mongo_uri)
    db = client["claims-fraud-db"]

    # Upload CSV data to MongoDB
    df = pd.read_csv(csv_path)
    csv_collection = db["claims_csv"]
    csv_collection.delete_many({})  # Remove previous entries
    csv_collection.insert_many(df.to_dict(orient="records"))

    # Upload model results to MongoDB
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    results_collection = db["model_results"]
    results_collection.delete_many({})
    results_collection.insert_one({"results": results})

@app.before_first_request
def upload_once():
    """
    On the first request, uploads CSV and results to MongoDB if MONGO_URI is set.
    This is only for archival/backup; the app does NOT serve data from MongoDB.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        csv_path = os.path.join(os.path.dirname(__file__), "motor_insurance_claims.csv")
        results_path = os.path.join(os.path.dirname(__file__), "model_results.pkl")
        upload_to_mongodb(csv_path, results_path, mongo_uri)
        print("✅ Data and results uploaded to MongoDB.")

@app.route("/")
def home():
    """
    Home route: loads results and a preview of the dataset from local files and renders them.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "motor_insurance_claims.csv")
    results_path = os.path.join(os.path.dirname(__file__), "model_results.pkl")

    # Load data and results from disk
    df = pd.read_csv(csv_path)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    # Display first few records as a preview
    data_preview = df.head().to_dict(orient="records")

    # Render your results page (replace with your template as needed)
    return render_template("results.html", results=results, data=data_preview)