import os
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from typing import List

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")

# Declare trained_features globally
trained_features: List[str] = []

def fetch_data() -> pd.DataFrame:
    """
    Fetch all insurance claim data from MongoDB (excluding _id field).

    Returns:
        pd.DataFrame: DataFrame containing all claims data from the database.
    """
    db = client["claims-fraud-db"]
    collection = db["motor_insurance_claims"]
    df = pd.DataFrame(list(collection.find({}, {"_id": 0})))  # Exclude `_id` for cleaner output
    return df

def initialize_models() -> None:
    """
    Preprocess data, train RandomForest and XGBoost models, and save the trained models.
    Also sets the global trained_features list corresponding to the model features.
    """
    global trained_features  # Ensure it's accessible

    df = fetch_data()
    df["fraud_reported"] = df["fraud_reported"].astype(str).str.strip().str.capitalize()
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)

    # Remove unnecessary columns
    df.drop(["umbrella_limit", "incident_hour_of_the_day", "number_of_vehicles", "fraud_label"], axis=1, inplace=True, errors="ignore")

    trained_features = df.select_dtypes(include=[np.number]).columns.tolist()

    X, y = df[trained_features], df["fraud_reported"].map(label_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Save trained models
    joblib.dump(rf_model, "random_forest.pkl")
    joblib.dump(xgb_model, "xgboost.pkl")

    print("✅ Models trained and saved!")


initialize_models()
