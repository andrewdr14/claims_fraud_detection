import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate Sample Data
np.random.seed(42)
data = {
    'claim_amount': np.random.randint(500, 20000, 500),
    'claim_history_count': np.random.randint(0, 10, 500),
    'customer_age': np.random.randint(18, 80, 500),
    'fraud_reported': np.random.choice([0, 1], size=500, p=[0.85, 0.15])  # 15% fraud cases
}

df = pd.DataFrame(data)

# Step 2: Split Data for Training & Testing
X = df[['claim_amount', 'claim_history_count', 'customer_age']]
y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the Fraud Detection Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Step 5: Test Model with a Sample Claim
sample_claim = np.array([[12000, 3, 45]])  # Claim Amount: £12,000, Past Claims: 3, Age: 45
fraud_prediction = model.predict(sample_claim)

print("Fraud Detected" if fraud_prediction[0] == 1 else "No Fraud Detected")