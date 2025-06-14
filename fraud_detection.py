import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. MySQL Connection
db_config = {
    'user': 'user',
    'password': 'password',
    'host': 'localhost',
    'database': 'insurance_data'
}

engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

# 2. Load Data
df = pd.read_sql("SELECT * FROM insurance_claims", con=engine)

# 3. Clean and Preprocess
df['fraud_reported'] = df['fraud_reported'].astype(str).str.upper().str.strip()
df['fraud_label'] = df['fraud_reported'].map({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
df = df[df['fraud_label'].isin([0, 1])]  # Drop anything unmapped

# 4. Feature Selection
features = [
    'age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
]

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features, inplace=True)

X = df[features]
y = df['fraud_label']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# 6. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix Plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Fraud Detection: Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0.5, 1.5], ['No Fraud', 'Fraud'])
plt.yticks([0.5, 1.5], ['No Fraud', 'Fraud'])
plt.tight_layout()
plt.show()

# 9. New Claim Prediction
sample_claim = np.array([[45, 210, 1000, 1400, 0, 0, 0, 8, 1, 1, 2, 5000, 3000, 1000, 1000]])
prediction = model.predict(sample_claim)
print("🔍 Prediction for Sample Claim:", "Fraudulent" if prediction[0] == 1 else "Legitimate")

# 10. Feature Importance Analysis
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
