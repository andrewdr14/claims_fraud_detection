import os
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
from typing import List

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

def fetch_data() -> pd.DataFrame:
    """
    Fetch all insurance claim data from MongoDB (excluding _id field).

    Returns:
        pd.DataFrame: DataFrame containing all claims data from the database.
    """
    db = client["claims-fraud-db"]
    collection = db["motor_insurance_claims"]
    df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Prepare features and target for training.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        feature_names (List[str]): Feature column names.
    """
    df = df.copy()
    df["fraud_reported"] = df["fraud_reported"].astype(str).str.strip().str.capitalize()
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)

    # Drop identifier and umbrella_limit (not used in rule, not predictive, not used by model)
    drop_cols = ["policy_number", "umbrella_limit"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # One-hot encode collision_type
    if "collision_type" in df.columns:
        df = pd.get_dummies(df, columns=["collision_type"], drop_first=True)

    # Build feature list: everything except the labels
    feature_names = [c for c in df.columns if c not in ["fraud_reported", "fraud_label"]]

    X = df[feature_names]
    y = df["fraud_label"]
    return X, y, feature_names

def train_and_save_models():
    """
    Trains Random Forest and XGBoost models and saves them along with the trained feature list.
    """
    df = fetch_data()
    X, y, feature_names = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    joblib.dump(rf_model, "random_forest.pkl")
    joblib.dump(xgb_model, "xgboost.pkl")
    with open("trained_features.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("✅ Models trained and saved!")

if __name__ == "__main__":
    train_and_save_models()