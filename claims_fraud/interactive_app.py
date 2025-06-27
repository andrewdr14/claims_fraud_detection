"""
Interactive Streamlit App for Comparing Custom Fraud Detection Modeling Workflows.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model import (
    split_data,
    feature_selection,
    train_xgboost,
    train_xgboost_gridsearch,
    train_catboost,
    train_catboost_gridsearch,
    evaluate_model
)

# Load dataset
df = pd.read_csv("cleaned_insurance_claims.csv")
target_col = "fraud_reported"
feature_cols = [col for col in df.columns if col != target_col]

# Constants
MODEL_OPTIONS = ["XGBoost", "CatBoost"]
FS_METHODS = ["Manual", "Random Forest", "L1 (Lasso)", "Auto (Best)"]
XGB_MODES = ["No", "Gridsearch"]

RF_THRESHOLDS = [
    ("Median (default)", "median"),
    ("Mean", "mean"),
    ("Top 25% (0.75*mean)", "0.75*mean"),
    ("Top 50% (0.5*mean)", "0.5*mean"),
    ("Top 75% (0.25*mean)", "0.25*mean"),
]
L1_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
MAX_WORKFLOWS = 4

# Info texts — now more descriptive
MODEL_INFOS = {
    "XGBoost": (
        "XGBoost is a powerful gradient boosting algorithm that builds predictive models "
        "by combining many weak learners (decision trees). It's highly effective for tabular data "
        "and often achieves strong performance in classification tasks like fraud detection."
    ),
    "CatBoost": (
        "CatBoost is a fast, scalable gradient boosting library designed to handle categorical features natively. "
        "It performs well on structured/tabular data and requires minimal hyperparameter tuning."
    )
}

FS_INFOS = {
    "Manual": (
        "Manually select which features you want to include in the model training process. "
        "This gives you complete control over feature engineering decisions."
    ),
    "Random Forest": (
        "Uses feature importance scores from a trained Random Forest to automatically select "
        "the most relevant features. You can adjust the importance threshold to keep only the top features."
    ),
    "L1 (Lasso)": (
        "Uses L1-penalized logistic regression to shrink unimportant feature coefficients to zero. "
        "This method performs automatic feature selection by eliminating irrelevant or redundant features."
    ),
    "Auto (Best)": (
        "Automatically tries all combinations of feature selection methods and hyperparameter settings, "
        "then returns the workflow configuration that yields the highest accuracy."
    )
}

XGB_MODES_INFOS = {
    "No": (
        "Train the model using default hyperparameters. This is fast but may not yield optimal results."
    ),
    "Gridsearch": (
        "Use grid search to explore a predefined space of hyperparameters and find the combination "
        "that gives the best model performance."
    )
}

RF_THRESHOLDS_INFOS = {
    label: f"Threshold: {value} — Keep features whose importance score exceeds this value." 
    for label, value in RF_THRESHOLDS
}

# Initialize session state
if "workflow_ids" not in st.session_state:
    st.session_state["workflow_ids"] = [1]


def next_wf_id() -> int:
    """Generates a unique ID for each new workflow."""
    if "next_wf_id" not in st.session_state:
        st.session_state["next_wf_id"] = 2
    val = st.session_state["next_wf_id"]
    st.session_state["next_wf_id"] += 1
    return val


# Visualization Functions
def render_classification_report(report_dict: dict, title: str = "Classification Report"):
    """Renders classification report as a styled DataFrame."""
    st.subheader(title)
    report_df = pd.DataFrame(report_dict).T
    st.dataframe(report_df.style.format(precision=2))


def plot_confusion_matrix(cm: np.ndarray, class_labels: list, title: str = "Confusion Matrix"):
    """Plots a confusion matrix using Seaborn heatmap."""
    st.subheader(title)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


def plot_importances(importances_df: pd.DataFrame, title: str = "Feature Importances"):
    """Plots feature importances as a horizontal bar chart."""
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, min(0.5 * len(importances_df), 8)))
    sns.barplot(data=importances_df, x="importance", y="feature", ax=ax, color="royalblue")
    ax.set_title(title)
    st.pyplot(fig)


# Workflow Configuration UI
def get_workflow_config(workflow_num: int, workflow_id: int, allow_remove: bool) -> dict:
    """
    Renders UI for configuring a single workflow.
    Returns a dictionary containing user-defined settings.
    """
    st.markdown(f"#### Workflow {workflow_num + 1} Settings")

    model_choice = st.selectbox("Training Model", MODEL_OPTIONS, key=f"model_{workflow_id}")
    st.info(MODEL_INFOS[model_choice])

    fs_method = st.selectbox("Feature Selection Method", FS_METHODS, key=f"fs_{workflow_id}")
    st.info(FS_INFOS[fs_method])

    config = {
        "model": model_choice,
        "fs_method": fs_method,
        "remove_clicked": False,
        "workflow_id": workflow_id
    }

    manual_disable = fs_method == "Auto (Best)"

    if fs_method == "Manual" and not manual_disable:
        config["features"] = st.multiselect(
            "Select Features", feature_cols, default=feature_cols, key=f"mf_{workflow_id}"
        )

    elif fs_method == "Random Forest" and not manual_disable:
        rf_label_to_value = {label: value for label, value in RF_THRESHOLDS}
        labels = [label for label, _ in RF_THRESHOLDS]
        selected = st.selectbox("Importance Threshold", labels, index=0, key=f"rfthresh_{workflow_id}")
        st.info(RF_THRESHOLDS_INFOS[selected])
        config["rf_threshold"] = rf_label_to_value[selected]

    elif fs_method == "L1 (Lasso)" and not manual_disable:
        alpha = st.number_input("L1 Regularization Strength (Alpha)", 0.0001, 1.0, 0.01, 0.01, key=f"l1alpha_{workflow_id}")
        st.info("Higher values remove more features.")
        config["l1_alpha"] = alpha

    if not manual_disable:
        mode = st.selectbox("Hyperparameter Tuning", XGB_MODES, index=0, key=f"ht_{workflow_id}")
        st.info(XGB_MODES_INFOS[mode])
        config["hypertune"] = mode

    if allow_remove:
        if st.button("Remove Workflow", key=f"remove_wf_{workflow_id}"):
            config["remove_clicked"] = True

    return config


# Run Single Workflow
def run_workflow(cfg: dict) -> dict:
    """
    Executes a single workflow based on its configuration.
    Returns evaluation metrics and model artifacts.
    """
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target_col)
    details = {}
    importances_df = None  # Always initialize to prevent UnboundLocalError

    # Feature Selection
    if cfg["fs_method"] == "Manual":
        if not cfg.get("features"):
            return {"error": "No features selected."}
        X_train, X_test, y_train, y_test = split_data(df, cfg["features"], target_col)
        X_train_fs, X_test_fs = X_train, X_test
        details["features"] = cfg["features"]

    elif cfg["fs_method"] in ["Random Forest", "L1 (Lasso)"]:
        method = "Random Forest" if cfg["fs_method"] == "Random Forest" else "L1"
        X_train_fs, X_test_fs, _, _, importances_df = feature_selection(
            method, X_train, y_train, X_test,
            threshold=cfg.get("rf_threshold"),
            alpha=cfg.get("l1_alpha")
        )
        details.update({k: v for k, v in cfg.items() if k in ["rf_threshold", "l1_alpha"]})

    else:
        X_train_fs, X_test_fs = X_train, X_test

    # Train Model
    if cfg["hypertune"] == "No":
        if cfg["model"] == "XGBoost":
            model, y_pred = train_xgboost(X_train_fs, y_train, X_test_fs)
        elif cfg["model"] == "CatBoost":
            model, y_pred = train_catboost(X_train_fs, y_train, X_test_fs)
        details["hypertune"] = "No"
    else:
        if cfg["model"] == "XGBoost":
            model, y_pred, _ = train_xgboost_gridsearch(X_train_fs, y_train, X_test_fs, y_test)
        elif cfg["model"] == "CatBoost":
            model, y_pred, _ = train_catboost_gridsearch(X_train_fs, y_train, X_test_fs, y_test)
        details["hypertune"] = "Gridsearch"

    # Evaluate
    report, cm, acc = evaluate_model(y_test, y_pred)

    return {
        "Accuracy": acc,
        "Report": report,
        "Confusion": cm,
        "Model": model,
        "Class Labels": model.classes_,
        "Importances": importances_df,
        "Details": details
    }


# Auto Best Mode: Try All Options
def run_auto_best() -> dict:
    """
    Tries all combinations of feature selection and hyperparameter tuning.
    Returns the best-performing configuration.
    """
    results = []

    # Try all options for both models
    for model_name in MODEL_OPTIONS:
        results.append(run_workflow({
            "fs_method": "Manual",
            "features": feature_cols,
            "hypertune": "No",
            "model": model_name
        }))
        results.append(run_workflow({
            "fs_method": "Manual",
            "features": feature_cols,
            "hypertune": "Gridsearch",
            "model": model_name
        }))

        for thresh in [v for _, v in RF_THRESHOLDS]:
            for mode in XGB_MODES:
                results.append(run_workflow({
                    "fs_method": "Random Forest",
                    "rf_threshold": thresh,
                    "hypertune": mode,
                    "model": model_name
                }))

        for alpha in L1_ALPHAS:
            for mode in XGB_MODES:
                results.append(run_workflow({
                    "fs_method": "L1 (Lasso)",
                    "l1_alpha": alpha,
                    "hypertune": mode,
                    "model": model_name
                }))

    try:
        best_result = max(results, key=lambda r: r.get("Accuracy", -1))
        best_result["AutoBest_Description"] = f"Auto (Best): {best_result['Details']}"
        return best_result
    except Exception as e:
        return {"error": f"Auto Best failed: {str(e)}"}


# Main Execution
st.title("Claims Fraud Detection: Compare Custom Workflows")
st.write("Set up and compare up to 4 custom modeling workflows side-by-side.")

# Add Workflow Button
add_col1, add_col2, add_col3 = st.columns([1, 2, 1])
with add_col2:
    if len(st.session_state["workflow_ids"]) < MAX_WORKFLOWS:
        if st.button("➕ Add Workflow", key="add_wf", use_container_width=True):
            st.session_state["workflow_ids"].append(next_wf_id())

# Manage Configurations
workflow_configs = []
remove_indices = []

for i, workflow_id in enumerate(st.session_state["workflow_ids"]):
    allow_remove = len(st.session_state["workflow_ids"]) > 1
    cfg = get_workflow_config(i, workflow_id, allow_remove)
    workflow_configs.append(cfg)
    if cfg["remove_clicked"]:
        remove_indices.append(i)
    st.markdown("---")

# Handle Removal
if remove_indices:
    for idx in sorted(remove_indices, reverse=True):
        del st.session_state["workflow_ids"][idx]
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# Run and Display Results
if st.button("Compare Workflows"):
    compare_results = []
    for i, cfg in enumerate(workflow_configs):
        with st.spinner(f"Running Workflow {i + 1}..."):
            if cfg["fs_method"] == "Auto (Best)":
                res = run_auto_best()
                if "error" in res:
                    st.error(f"Workflow {i + 1} error: {res['error']}")
                    continue
            else:
                res = run_workflow(cfg)
                if "error" in res:
                    st.error(f"Workflow {i + 1} error: {res['error']}")
                    continue
            compare_results.append(res)

    tabs = st.tabs([f"Workflow {i + 1}" for i in range(len(compare_results))])
    for i, res in enumerate(compare_results):
        with tabs[i]:
            if "AutoBest_Description" in res:
                st.info(res["AutoBest_Description"])
            st.write(f"**{res['Model']}**")
            st.write(f"**Accuracy:** {res['Accuracy']:.3f}")
            render_classification_report(res["Report"])
            plot_confusion_matrix(res["Confusion"], res["Class Labels"])
            if res["Importances"] is not None:
                plot_importances(res["Importances"])