import os
import joblib
import pickle
import pandas as pd
from flask import Flask, render_template, send_file, abort
from dotenv import load_dotenv
from . import model  # If running as a module/package, otherwise use import model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for plotting
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# ----------------------------------------------------------------------
# DATA DESCRIPTIONS AND FRAUD RULE TEXT
# ----------------------------------------------------------------------

DATA_DESCRIPTIONS = {
    "policy_number": "Unique identifier for each insurance policy (e.g. 'POL12345').",
    "policy_deductible": "Out-of-pocket amount the policyholder must pay before insurance covers a claim (500, 1000, or 2000).",
    "policy_annual_premium": "Yearly cost of the insurance policy (500 to 3000).",
    "umbrella_limit": "Maximum coverage provided by umbrella insurance (50,000, 100,000, or 150,000).",
    "insured_age": "Age of the insured individual (18 to 85).",
    "incident_hour_of_the_day": "Hour when the incident occurred (0 to 23).",
    "collision_type": "Type of accident ('Rear-End', 'Side Impact', etc).",
    "number_of_vehicles": "Number of vehicles involved in the incident (1 to 5).",
    "total_claim_amount": "Total amount claimed for the incident (1,000 to 20,000).",
    "fraud_reported": "Whether the claim was reported as fraud ('Yes' or 'No')."
}

# ----------------------------------------------------------------------
# FLASK APP INITIALIZATION
# ----------------------------------------------------------------------

# Set template_folder to find templates in the project root (one level above this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ----------------------------------------------------------------------
# LOAD TRAINED MODELS AND FEATURES
# ----------------------------------------------------------------------

# Load the list of features the models were trained on
with open("trained_features.pkl", "rb") as f:
    trained_features = pickle.load(f)

# Load trained models from disk
rf_model = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost.pkl")

# ----------------------------------------------------------------------
# HELPER FUNCTIONS FOR PLOTS
# ----------------------------------------------------------------------

def plot_feature_importances(rf_model, xgb_model, feature_names):
    """
    Plot a side-by-side comparison of feature importances for RF and XGB.
    Returns base64-encoded PNG.
    """
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_

    # Sort by mean importance for visual clarity
    mean_imp = (rf_imp + xgb_imp) / 2
    sorted_idx = np.argsort(mean_imp)[::-1]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    indices = np.arange(len(feature_names))

    plt.bar(indices - bar_width/2, rf_imp[sorted_idx], bar_width, label="Random Forest", color='#007bff', alpha=0.85)
    plt.bar(indices + bar_width/2, xgb_imp[sorted_idx], bar_width, label="XGBoost", color='#ff9900', alpha=0.7)
    plt.xticks(indices, np.array(feature_names)[sorted_idx], rotation=45, ha='right', fontsize=10)
    plt.ylabel("Importance")
    plt.title("Feature Importance Comparison (Random Forest vs XGBoost)")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64

def plot_confusion_matrix(cm, title):
    """
    Render a confusion matrix as a PNG image and return its base64-encoded string.
    """
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["No fraud", "Fraud"])
    plt.yticks(tick_marks, ["No fraud", "Fraud"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i, j in [(i, j) for i in range(2) for j in range(2)]:
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_roc_curve(y_true, y_score, title):
    """
    Render a ROC curve as a PNG image and return its base64-encoded string.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc_score(y_true, y_score))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_pr_curve(y_true, y_score, title):
    """
    Render a Precision-Recall curve as a PNG image and return its base64-encoded string.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label="PR curve (AP = %0.2f)" % ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# ----------------------------------------------------------------------
# MAIN ROUTES
# ----------------------------------------------------------------------

@app.route("/")
def evaluation() -> str:
    """
    Main dashboard route.
    Fetches data, processes it, applies models, and renders the results template.
    """
    # Fetch claims data from the database
    df = model.fetch_data()

    # Filter columns for display
    all_columns = list(df.columns)
    all_columns = [col for col in all_columns if col in DATA_DESCRIPTIONS]
    used_for_training = trained_features

    # Ensure the fraud label is normalized for all rows
    df["fraud_reported"] = df["fraud_reported"].astype(str).str.strip().str.capitalize()
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)

    # ------------------------------------------------------------------
    # ENSURE CONSISTENT ONE-HOT ENCODING FOR COLLISION_TYPE
    # ------------------------------------------------------------------
    # One-hot encode collision_type to match training
    if "collision_type" in df.columns:
        df = pd.get_dummies(df, columns=["collision_type"], drop_first=True)

    # Add any missing dummy columns that were present during training
    for col in trained_features:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with zeros

    # Reorder columns to match training
    X_for_pred = df[trained_features]

    # ------------------------------------------------------------------
    # MODEL PREDICTION AND EVALUATION
    # ------------------------------------------------------------------
    y_true = df["fraud_label"]
    y_pred_rf = rf_model.predict(X_for_pred)
    y_pred_xgb = xgb_model.predict(X_for_pred)

    # Classification reports (dict for easier template rendering)
    rf_results = classification_report(y_true, y_pred_rf, output_dict=True)
    xgb_results = classification_report(y_true, y_pred_xgb, output_dict=True)

    # Confusion matrices
    rf_cm = confusion_matrix(y_true, y_pred_rf)
    xgb_cm = confusion_matrix(y_true, y_pred_xgb)

    # ROC-AUC and PR-AUC
    rf_proba = rf_model.predict_proba(X_for_pred)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_for_pred)[:, 1]
    rf_roc_auc = roc_auc_score(y_true, rf_proba)
    xgb_roc_auc = roc_auc_score(y_true, xgb_proba)
    rf_pr_auc = average_precision_score(y_true, rf_proba)
    xgb_pr_auc = average_precision_score(y_true, xgb_proba)

    # Generate images for confusion matrices and curves
    rf_cm_img = plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
    xgb_cm_img = plot_confusion_matrix(xgb_cm, "XGBoost Confusion Matrix")
    rf_roc_img = plot_roc_curve(y_true, rf_proba, "Random Forest ROC Curve")
    xgb_roc_img = plot_roc_curve(y_true, xgb_proba, "XGBoost ROC Curve")
    rf_pr_img = plot_pr_curve(y_true, rf_proba, "Random Forest Precision-Recall Curve")
    xgb_pr_img = plot_pr_curve(y_true, xgb_proba, "XGBoost Precision-Recall Curve")
    feature_importance_img = plot_feature_importances(rf_model, xgb_model, trained_features)
    
    # ------------------------------------------------------------------
    # SUMMARY STATISTICS FOR TEMPLATES
    # ------------------------------------------------------------------
    policy_holder_count = len(df)
    fraud_cases = (df["fraud_reported"] == "Yes").sum()
    non_fraud_cases = (df["fraud_reported"] == "No").sum()

    # Compute detailed statistics for numeric columns (excluding a few)
    excluded_columns = ["umbrella_limit", "incident_hour_of_the_day", "number_of_vehicles", "fraud_label"]
    numeric_cols = [col for col in used_for_training if col not in excluded_columns]

    summary_stats: Dict[str, Any] = {
        "Total Policy Holders": policy_holder_count,
        "Fraud Reported": fraud_cases,
        "No Fraud Reported": non_fraud_cases
    }
    for col in numeric_cols:
        summary_stats[f"{col} - Min"] = df[col].min()
        summary_stats[f"{col} - Max"] = df[col].max()
        summary_stats[f"{col} - Mean"] = round(df[col].mean(), 2)
        summary_stats[f"{col} - Median"] = df[col].median()
        summary_stats[f"{col} - Std Dev"] = round(df[col].std(), 2)

    # ------------------------------------------------------------------
    # RENDER TEMPLATE WITH ALL DATA AND VISUALIZATIONS
    # ------------------------------------------------------------------
    return render_template(
        "results.html",
        rf_results=rf_results,
        xgb_results=xgb_results,
        summary_stats=summary_stats,
        rf_cm=rf_cm,
        xgb_cm=xgb_cm,
        rf_roc_auc=rf_roc_auc,
        xgb_roc_auc=xgb_roc_auc,
        rf_pr_auc=rf_pr_auc,
        xgb_pr_auc=xgb_pr_auc,
        rf_cm_img=rf_cm_img,
        xgb_cm_img=xgb_cm_img,
        rf_roc_img=rf_roc_img,
        xgb_roc_img=xgb_roc_img,
        rf_pr_img=rf_pr_img,
        xgb_pr_img=xgb_pr_img,
        data_descriptions=DATA_DESCRIPTIONS,
        all_columns=all_columns,
        used_for_training=used_for_training,
        feature_importance_img=feature_importance_img
    )

@app.route("/download-data")
def download_data() -> Any:
    """
    Allow users to download the dataset as a CSV file.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "motor_insurance_claims.csv")
    if not os.path.exists(csv_path):
        abort(404, description="Dataset file not found. Please generate it first.")
    return send_file(csv_path, as_attachment=True)

# ----------------------------------------------------------------------
# RUN THE APP
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Dynamically assign Render's port for deployment or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
