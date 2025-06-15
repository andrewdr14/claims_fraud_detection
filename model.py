import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]
collection = db["motor_insurance_claims"]

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model_is_trained = False
trained_features = []  # Dynamically filled

def fetch_data():
    """Retrieve claim data from MongoDB Atlas."""
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id field
    return pd.DataFrame(data)

def initialize_model():
    """Train model using data from MongoDB."""
    global model_is_trained, trained_features
    df = fetch_data()
    df = _clean_columns(df)
    
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)
    
    df = df[df["fraud_label"].isin([0, 1])]
    df.dropna(axis=1, how="all", inplace=True)
    
    available = df.select_dtypes(include=[np.number]).columns.tolist()
    available.remove("fraud_label")
    
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=available, inplace=True)
    
    trained_features = available
    
    X, y = df[trained_features], df["fraud_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    model_is_trained = True
    
    print("✅ Model trained on features:", trained_features)
    print(classification_report(y_test, model.predict(X_test)))

def predict_fraud():
    """Score new claims using the trained model."""
    df = fetch_data()
    df = _clean_columns(df)
    
    df[trained_features] = df[trained_features].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=trained_features, inplace=True)
    
    df["Fraud Probability"] = model.predict_proba(df[trained_features])[:, 1]
    return df

def _clean_columns(df):
    """Clean column names for consistency."""
    df.columns = df.columns.str.strip().str.replace("-", "_", regex=False)
    return df

# Execute model training
if __name__ == "__main__":
    initialize_model()