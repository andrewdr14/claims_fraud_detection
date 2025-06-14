import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

# 1️⃣ Load Environment Variables
load_dotenv()

# Assign database credentials from .env file
db_config = {
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'database': os.getenv("DB_NAME")
}

# Debugging: Ensure environment variables are loaded correctly
print("🔍 DB Config:", db_config)

# Raise an error if any variable is missing
if None in db_config.values():
    raise ValueError("❌ ERROR: One or more environment variables are missing. Check your .env file.")

# 2️⃣ Establish MySQL Connection
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

# 3️⃣ Load Data from MySQL
df = pd.read_sql("SELECT * FROM insurance_claims", con=engine)

# 4️⃣ Data Cleaning & Preprocessing
df['fraud_reported'] = df['fraud_reported'].astype(str).str.upper().str.strip()
df['fraud_label'] = df['fraud_reported'].map({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
df = df[df['fraud_label'].isin([0, 1])]  # Remove unmapped values

# 5️⃣ Feature Selection
features = [
    'age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
]

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features, inplace=True)

# Define feature matrix (X) and target variable (y)
X = df[features]
y = df['fraud_label']

# 6️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# 7️⃣ Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Model Evaluation
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# 9️⃣ Confusion Matrix Visualization

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define TP, FP, TN, FN labels separately
tp, fp, fn, tn = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]

# Plot confusion matrix
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])

# Add TP, FP, TN, FN labels **outside** the heatmap
plt.text(0, 0.1, "TN", color="black", fontsize=12, va="center")
plt.text(0, 1.1, "FN", color="black", fontsize=12, va="center")
plt.text(1, 0.1, "FP", color="black", fontsize=12, va="center")
plt.text(1, 1.1, "TP", color="black", fontsize=12, va="center")

# Formatting
plt.title("Fraud Detection: Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.xticks([0.5, 1.5], ['No Fraud', 'Fraud'])
plt.yticks([0.5, 1.5], ['No Fraud', 'Fraud'])
plt.tight_layout()
plt.show()

# 🔟 Sample Prediction (New Claim)
sample_claim = np.array([[45, 210, 1000, 1400, 0, 0, 0, 8, 1, 1, 2, 5000, 3000, 1000, 1000]])
prediction = model.predict(sample_claim)
print("🔍 Prediction for Sample Claim:", "Fraudulent" if prediction[0] == 1 else "Legitimate")

# 1️⃣1️⃣ Feature Importance Analysis
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Top Predictive Features:")
print(importance_df.head(10))

# Optional: Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title("Top 10 Features Predicting Fraud")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

def predict_fraud(df):
    """Predict fraud probabilities for a given dataset."""
    features = [
        "age", "months_as_customer", "policy_deductable", "policy_annual_premium",
        "umbrella_limit", "capital_gains", "capital_loss", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "bodily_injuries", "witnesses",
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"
    ]

    X = df[features]
    fraud_probs = model.predict_proba(X)[:, 1]  # Get fraud probability (column 1)
    return fraud_probs






