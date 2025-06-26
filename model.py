import os
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")  # Suppress all warnings for clean output

# ========================== DATA & MONGO FUNCTIONS ==========================
def load_data(raw_path, clean_path):
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"{clean_path} not found. Please run the Jupyter notebook first to generate the cleaned data."
        )
    df_raw = pd.read_csv(raw_path)
    df_clean = pd.read_csv(clean_path)
    return df_raw, df_clean

def save_to_mongodb(df: pd.DataFrame, collection_name: str, client=None):
    if client is None:
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
    db = client["claims-fraud-db"]
    collection = db[collection_name]
    collection.delete_many({})
    if not df.empty:
        collection.insert_many(df.to_dict("records"))

def load_categorical_mappings(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"{mapping_path} not found. Please extract categorical mappings from the Jupyter notebook and save as a pickle."
        )
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    return mappings

def decode_categorical_column(series, mapping):
    return series.map(mapping)

# ========================== RANDOM FOREST FEATURE SELECTION ==========================
def random_forest_feature_selection(X_train, y_train, X_test, threshold="median"):
    rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_fs.fit(X_train, y_train)
    selector = SelectFromModel(rf_fs, prefit=True, threshold=threshold)
    X_train_fs = selector.transform(X_train)
    X_test_fs = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    return rf_fs, selector, X_train_fs, X_test_fs, selected_features

def get_omitted_features(X, selected_features):
    return [col for col in X.columns if col not in selected_features]

# ========================== XGBOOST TRAINING AND TUNING ==========================
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred

def grid_search_xgboost(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1,
        cv=3,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    return best_xgb, y_pred, grid_search.best_params_

# ========================== VISUALIZATIONS ==========================
def visualize_all(
    df_clean,
    selected_features,
    rf_fs,
    omitted_features,
    y_test,
    y_pred_default,
    xgb_default,
    y_pred_grid,
    best_xgb,
    mappings
):
    # Feature importances
    importances = rf_fs.feature_importances_[[list(df_clean.drop('fraud_reported', axis=1).columns).index(f) for f in selected_features]]
    plt.figure(figsize=(10, 6))
    plt.barh(selected_features, importances)
    plt.xlabel("Feature Importance")
    plt.title("Selected Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig("rf_selected_feature_importances.png")
    plt.close()

    # Save omitted features as txt for Streamlit
    with open("omitted_features.txt", "w") as f:
        for feat in omitted_features:
            f.write(f"{feat}\n")

    # Confusion matrix: XGBoost default
    plot_and_save_confusion_matrix(
        y_test, y_pred_default,
        title="Confusion Matrix (XGBoost Default)",
        filename="confusion_matrix_default.png",
        labels=decode_labels(xgb_default.classes_, mappings.get('fraud_reported', None))
    )

    # Confusion matrix: XGBoost grid search
    plot_and_save_confusion_matrix(
        y_test, y_pred_grid,
        title="Confusion Matrix (XGBoost GridSearch)",
        filename="confusion_matrix_gridsearch.png",
        labels=decode_labels(best_xgb.classes_, mappings.get('fraud_reported', None))
    )

    # Pie charts for fraud allocation
    plot_fraud_pie_charts(df_clean, mappings, fraud_col='fraud_reported')

def decode_labels(labels, mapping):
    if mapping is None:
        return labels
    return [mapping.get(l, l) for l in labels]

def plot_and_save_confusion_matrix(y_true, y_pred, title, filename, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=None if labels is None else range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def plot_fraud_pie_charts(df, mappings, fraud_col='fraud_reported'):
    fraud_df = df[df[fraud_col] == 1].copy()

    # Incident severity
    if 'incident_severity' in fraud_df.columns:
        sev_map = mappings.get('incident_severity', None)
        labels = None
        if sev_map is not None:
            fraud_df['incident_severity_decoded'] = decode_categorical_column(fraud_df['incident_severity'], sev_map)
            counts = fraud_df['incident_severity_decoded'].value_counts()
            labels = counts.index
        else:
            counts = fraud_df['incident_severity'].value_counts()
            labels = counts.index

        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            counts, labels=None, autopct='%1.1f%%', startangle=140
        )
        plt.title('Fraudulent Claims Allocation by Incident Severity')
        plt.legend(wedges, labels, title="Incident Severity", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("fraud_pie_incident_severity.png")
        plt.close()
    else:
        print("incident_severity column not found in data.")

    # Insured hobbies
    if 'insured_hobbies' in fraud_df.columns:
        hob_map = mappings.get('insured_hobbies', None)
        if hob_map is not None:
            fraud_df['insured_hobbies_decoded'] = decode_categorical_column(fraud_df['insured_hobbies'], hob_map)
            counts = fraud_df['insured_hobbies_decoded'].value_counts()
        else:
            counts = fraud_df['insured_hobbies'].value_counts()

        total = counts.sum()
        percentages = (counts / total) * 100
        large_mask = percentages > 4.1
        small_mask = ~large_mask
        counts_large = counts[large_mask]
        counts_small_sum = counts[small_mask].sum()

        labels = list(counts_large.index)
        sizes = list(counts_large.values)

        if counts_small_sum > 0:
            labels.append("Others")
            sizes.append(counts_small_sum)

        plt.figure(figsize=(7, 7))
        wedges, texts, autotexts = plt.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=140
        )
        plt.title('Fraudulent Claims Allocation by Insured Hobbies')
        plt.legend(wedges, labels, title="Insured Hobby", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("fraud_pie_insured_hobbies.png")
        plt.close()
    else:
        print("insured_hobbies column not found in data.")

# ========================== MAIN PIPELINE ==========================
def main():
    raw_path = "insurance_claims.csv"
    clean_path = "cleaned_insurance_claims.csv"
    mapping_path = "categorical_mappings.pkl"
    df_raw, df_clean = load_data(raw_path, clean_path)
    mappings = load_categorical_mappings(mapping_path)

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    save_to_mongodb(df_raw, "motor_insurance_claims_raw", client=client)
    save_to_mongodb(df_clean, "motor_insurance_claims_clean", client=client)

    if 'fraud_reported' not in df_clean.columns:
        raise ValueError("Cleaned data must contain a 'fraud_reported' column.")

    X = df_clean.drop('fraud_reported', axis=1)
    y = df_clean['fraud_reported']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    rf_fs, selector, X_train_fs, X_test_fs, selected_features = random_forest_feature_selection(
        X_train, y_train, X_test, threshold="median"
    )
    omitted_features = get_omitted_features(X, selected_features)

    xgb_default, y_pred_default = train_xgboost(X_train_fs, y_train, X_test_fs, y_test)
    best_xgb, y_pred_grid, best_params = grid_search_xgboost(X_train_fs, y_train, X_test_fs, y_test)

    # Save model artifacts for Streamlit
    joblib.dump(rf_fs, "random_forest_feature_selector.pkl")
    joblib.dump(best_xgb, "xgboost_final.pkl")
    with open("selected_features.pkl", "wb") as f:
        pickle.dump(list(selected_features), f)
    with open("best_xgboost_params.txt", "w") as f:
        f.write(str(best_params))

    # Save classification reports
    report_default = classification_report(y_test, y_pred_default)
    report_grid = classification_report(y_test, y_pred_grid)
    with open("classification_report_default.txt", "w") as f:
        f.write(report_default)
    with open("classification_report_gridsearch.txt", "w") as f:
        f.write(f"Best GridSearchCV Params:\n{best_params}\n\n{report_grid}")

    visualize_all(
        df_clean,
        selected_features,
        rf_fs,
        omitted_features,
        y_test,
        y_pred_default,
        xgb_default,
        y_pred_grid,
        best_xgb,
        mappings
    )

if __name__ == "__main__":
    main()