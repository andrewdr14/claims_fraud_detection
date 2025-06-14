import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
from dotenv import load_dotenv

# Load env vars and connect to MongoDB
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]
collection = db["insurance_claims"]

# Global model and lazy training flag
model = RandomForestClassifier(n_estimators=100, random_state=42)
model_is_trained = False

features = [
    'age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
]

def initialize_model():
    global model_is_trained
    if not model_is_trained:
        train_model()
        model_is_trained = True

def train_model():
    df = pd.DataFrame(list(collection.find()))
    df.columns = df.columns.str.strip().str.replace("-", "_", regex=False)
    df['fraud_reported'] = df['fraud_reported'].astype(str).str.upper().str.strip()
    df['fraud_label'] = df['fraud_reported'].map({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
    df = df[df['fraud_label'].isin([0, 1])]
    available = [col for col in features if col in df.columns]
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=available, inplace=True)
    X, y = df[available], df['fraud_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))

def predict_fraud(df):
    initialize_model()
    df.columns = df.columns.str.strip().str.replace("-", "_", regex=False)
    available = [col for col in features if col in df.columns]
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=available, inplace=True)
    df["Fraud Probability"] = model.predict_proba(df[available])[:, 1]
    return df  # Return enriched dataframe—no DB write here