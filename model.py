import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=100, random_state=42)
model_is_trained = False
trained_features = []  # Dynamically filled

def initialize_model(df):
    global model_is_trained, trained_features
    if not model_is_trained:
        df = _clean_columns(df)
        label_map = {'Y': 1, 'N': 0, 'YES': 1, 'NO': 0}
        df['fraud_reported'] = df['fraud_reported'].astype(str).str.upper().str.strip()
        df['fraud_label'] = df['fraud_reported'].map(label_map)
        df = df[df['fraud_label'].isin([0, 1])]
        df = df.dropna(axis=1, how='all')
        available = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'fraud_label' in available:
            available.remove('fraud_label')
        df[available] = df[available].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=available, inplace=True)
        trained_features = available
        X, y = df[available], df['fraud_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        model_is_trained = True
        print("✅ Model trained on features:", trained_features)
        print(classification_report(y_test, model.predict(X_test)))

def predict_fraud(df):
    df = _clean_columns(df)
    df[trained_features] = df[trained_features].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=trained_features, inplace=True)
    df["Fraud Probability"] = model.predict_proba(df[trained_features])[:, 1]
    return df

def _clean_columns(df):
    df.columns = df.columns.str.strip().str.replace("-", "_", regex=False)
    return df