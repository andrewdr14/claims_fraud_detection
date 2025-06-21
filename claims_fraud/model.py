"""
model.py

This script trains a RandomForestClassifier using the generated CSV data
and saves both the trained model and its results to disk.
It does NOT interact with MongoDB.

Usage:
    python claims_fraud/model.py
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(csv_path):
    """
    Trains a RandomForestClassifier on the insurance claims dataset.

    Args:
        csv_path (str): Path to the CSV data.

    Returns:
        tuple: (trained model, dict of results/metrics)
    """
    # Load data
    df = pd.read_csv(csv_path)
    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Evaluate model
    score = model.score(X, y)
    results = {"accuracy": score}

    return model, results

if __name__ == "__main__":
    # Paths for input and output files
    csv_path = os.path.join(os.path.dirname(__file__), "motor_insurance_claims.csv")
    model_path = os.path.join(os.path.dirname(__file__), "random_forest.pkl")
    results_path = os.path.join(os.path.dirname(__file__), "model_results.pkl")

    # Train model and get results
    model, results = train_model(csv_path)

    # Save the trained model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save results/metrics
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"✅ Model trained and saved to {model_path}! Accuracy: {results['accuracy']:.3f}")
    print(f"✅ Results saved to {results_path}")