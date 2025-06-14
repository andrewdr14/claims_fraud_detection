import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 1️⃣ Load Environment Variables
load_dotenv()

# 2️⃣ Get MongoDB URI from Environment Variables
mongo_uri = os.getenv("MONGO_URI")  # Ensure this is correctly set in Render

# 3️⃣ Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]  # Use the correct database name
collection = db["insurance_claims"]  # Use the correct collection name

# 2️⃣ Load Data from MongoDB
df = pd.DataFrame(list(collection.find()))  # Load claims data from MongoDB

# 3️⃣ Data Cleaning & Preprocessing
df['fraud_reported'] = df['fraud_reported'].astype(str).str.upper().str.strip()
df['fraud_label'] = df['fraud_reported'].map({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
df = df[df['fraud_label'].isin([0, 1])]  # Remove unmapped values

# 4️⃣ Feature Selection
features = [
    'age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
]


df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features, inplace=True)

# Define feature matrix (X) and target variable (y)
X = df[features]
y = df['fraud_label']

# 5️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# 6️⃣ Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Model Evaluation
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# 8️⃣ Store Fraud Predictions in MongoDB
def store_prediction(data):
    """Store fraud predictions in MongoDB."""
    collection.insert_one(data)

def predict_fraud(df):
    """Predict fraud probabilities for a given dataset."""
    X = df[features]
    fraud_probs = model.predict_proba(X)[:, 1]  # Get fraud probability (column 1)
    
    # Store predictions in MongoDB
    df["Fraud Probability"] = fraud_probs
    store_prediction(df.to_dict(orient="records"))
    
    return fraud_probs