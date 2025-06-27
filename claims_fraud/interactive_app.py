"""
Interactive Streamlit App for Comparing Custom Fraud Detection Modeling Workflows.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import (
    split_data,
    rf_feature_selection,
    l1_feature_selection,
    train_xgboost,
    train_xgboost_gridsearch,
    evaluate_model
)

# Load dataset
df = pd.read_csv("cleaned_insurance_claims.csv")
target_col = "fraud_reported"
feature_cols = [col for col in df.columns if col != target_col]

# Constants
MODEL_OPTIONS = ["XGBoost"]
MODEL_INFOS = {
    "XGBoost": (
        "XGBoost is a powerful, efficient gradient boosting algorithm well-suited "
        "for tabular data. It uses gradient boosting, an ensemble technique that builds "
        "strong predictive models from many weak learners (decision trees)."
    )
}

FS_METHODS = ["Manual", "Random Forest", "L1 (Lasso)", "Auto (Best)"]
FS_INFOS = {
    "Manual": (
        "Manually select which features to use for training. This gives you full control "
        "over which variables are included in the model."
    ),
    "Random Forest": (
        "Uses feature importances from a random forest to select the most relevant features. "
        "You can set a threshold for which features to keep."
    ),
    "L1 (Lasso)": (
        "Uses L1-penalized logistic regression to automatically zero out less important features. "
        "The alpha parameter controls the strength of the penalty."
    ),
    "Auto (Best)": (
        "Automatically tries all options for feature selection and hyperparameter tuning, "
        "then uses the combination that results in the highest accuracy."
    ),
}

XGB_MODES = ["No", "Gridsearch"]
XGB_MODES_INFOS = {
    "No": "Train the model with default hyperparameters.",
    "Gridsearch": "Gridsearch will search over a predefined set of hyperparameters to try to find the best combination for your data.",
}

RF_THRESHOLDS = [
    ("Median (default)", "median"),
    ("Mean", "mean"),
    ("Top 25% (0.75*mean)", "0.75*mean"),
    ("Top 50% (0.5*mean)", "0.5*mean"),
    ("Top 75% (0.25*mean)", "0.25*mean"),
]
RF_THRESHOLDS_INFOS = {
    "median": "Select features whose importance is above the median importance from the random forest.",
    "mean": "Select features above the mean importance from the random forest.",
    "0.75*mean": "Select only the most important features (top 25%).",
    "0.5*mean": "Select the top 50% of features by importance.",
    "0.25*mean": "Select the top 75% of features by importance.",
}

L1_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
MAX_WORKFLOWS = 4

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

# UI Title and Instructions
st.title("Claims Fraud Detection: Compare Custom Workflows")
st.write(
    "Set up and compare up to 4 custom modeling workflows side-by-side. "
    "Hover over the info icons or see the blue boxes for more details."
)

# Add Workflow Button
add_col1, add_col2, add_col3 = st.columns([1, 2, 1])
with add_col2:
    if len(st.session_state["workflow_ids"]) < MAX_WORKFLOWS:
        if st.button("➕ Add Workflow", key="add_wf", use_container_width=True):
            st.session_state["workflow_ids"].append(next_wf_id())

# Visualization Functions
def render_classification_report(report_dict: dict, title: str = "Classification Report") -> None:
    """Renders classification report as a styled DataFrame."""
    st.subheader(title)
    report_df = pd.DataFrame(report_dict).T
    st.dataframe(report_df.style.format(precision=2))

def plot_confusion_matrix(cm: list, class_labels: list, title: str = "Confusion Matrix") -> None:
    """Plots a confusion matrix using Seaborn heatmap."""
    st.subheader(title)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def plot_importances(importances_df: pd.DataFrame, title: str = "Feature Importances") -> None:
    """Plots feature importances as a horizontal bar chart."""
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, min(0.5 * len(importances_df), 8)))
    sns.barplot(data=importances_df, x="importance", y="feature", ax=ax, color="royalblue")
    ax.set_title(title)
    st.pyplot(fig)

# Configuration UI per Workflow
def get_workflow_config(workflow_num: int, workflow_id: int, allow_remove: bool) -> dict:
    """
    Renders UI for configuring a single workflow.

    Returns
    -------
    dict
        Dictionary containing the user-defined configuration settings.
    """
    st.markdown(f"#### Workflow {workflow_num + 1} Settings")

    model_choice = st.selectbox(
        f"Workflow {workflow_num + 1}: Training model",
        MODEL_OPTIONS,
        key=f"model_{workflow_id}",
        help="Select the machine learning model to train."
    )
    st.info(MODEL_INFOS[model_choice])

    fs_choice = st.selectbox(
        f"Workflow {workflow_num + 1}: Feature selection method",
        FS_METHODS,
        key=f"fs_{workflow_id}",
        help="Choose how features will be selected for the model."
    )
    st.info(FS_INFOS[fs_choice])

    features_to_use = feature_cols
    rf_threshold = RF_THRESHOLDS[0][1]
    l1_alpha = 0.01
    hypertune_choice = XGB_MODES[0]
    manual_disable = (fs_choice == "Auto (Best)")

    # Manual Feature Selection
    if fs_choice == "Manual" and not manual_disable:
        features_to_use = st.multiselect(
            f"Workflow {workflow_num + 1}: Select features to keep",
            feature_cols, default=feature_cols, key=f"mf_{workflow_id}",
            help="Choose which features to include in training."
        )

    # Random Forest Feature Selection
    elif fs_choice == "Random Forest" and not manual_disable:
        rf_labels = [label for label, _ in RF_THRESHOLDS]
        rf_label_to_value = {label: value for label, value in RF_THRESHOLDS}
        rf_label = st.selectbox(
            f"Workflow {workflow_num + 1}: RF feature importance threshold",
            rf_labels, index=0, key=f"rfthresh_{workflow_id}",
            help="Threshold for which features are considered important enough to keep."
        )
        st.info(RF_THRESHOLDS_INFOS[rf_label_to_value[rf_label]])
        rf_threshold = rf_label_to_value[rf_label]

    # L1 Feature Selection
    elif fs_choice == "L1 (Lasso)" and not manual_disable:
        l1_alpha = st.number_input(
            f"Workflow {workflow_num + 1}: L1 regularization strength (alpha)",
            min_value=0.0001, max_value=1.0, value=0.01, step=0.01, key=f"l1alpha_{workflow_id}",
            help="The regularization strength for L1 (Lasso). Higher values remove more features."
        )
        st.info("L1 alpha controls the strength of the penalty. Higher values will result in more features being zeroed out.")

    # Hyperparameter Tuning Option
    if not manual_disable:
        hypertune_choice = st.selectbox(
            f"Workflow {workflow_num + 1}: Use hyperparameter tuning?",
            XGB_MODES, index=0, key=f"ht_{workflow_id}",
            help="Choose whether to tune the model's hyperparameters using gridsearch."
        )
        st.info(XGB_MODES_INFOS[hypertune_choice])

    # Remove Button
    remove_clicked = False
    if allow_remove:
        if st.button("Remove Workflow", key=f"remove_wf_{workflow_id}"):
            remove_clicked = True

    return {
        "model": model_choice,
        "fs_method": fs_choice,
        "features": features_to_use,
        "rf_threshold": rf_threshold,
        "l1_alpha": l1_alpha,
        "hypertune": hypertune_choice,
        "remove_clicked": remove_clicked,
        "workflow_id": workflow_id
    }

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

# Run Single Workflow
def run_workflow(cfg: dict) -> dict:
    """
    Executes a single workflow based on its configuration.

    Parameters
    ----------
    cfg : dict
        A dictionary containing workflow configuration settings.

    Returns
    -------
    dict
        Evaluation metrics and model artifacts.
    """
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target_col)
    importances_df = None
    details = {}

    # Feature Selection Logic
    if cfg["fs_method"] == "Manual":
        if not cfg["features"]:
            return {"error": "No features selected."}
        X_train, X_test, y_train, y_test = split_data(df, cfg["features"], target_col)
        X_train_fs, X_test_fs = X_train, X_test
        details["features"] = cfg["features"]

    elif cfg["fs_method"] == "Random Forest":
        X_train_fs, X_test_fs, _, _, importances_df = rf_feature_selection(
            X_train, y_train, X_test, threshold=cfg["rf_threshold"]
        )
        details["rf_threshold"] = cfg["rf_threshold"]

    elif cfg["fs_method"] == "L1 (Lasso)":
        X_train_fs, X_test_fs, _, _, importances_df = l1_feature_selection(
            X_train, y_train, X_test, alpha=cfg["l1_alpha"]
        )
        details["l1_alpha"] = cfg["l1_alpha"]

    else:
        X_train_fs, X_test_fs = X_train, X_test

    # Model Training
    if cfg["hypertune"] == "No":
        model, y_pred = train_xgboost(X_train_fs, y_train, X_test_fs)
        details["hypertune"] = "No"
    else:
        model, y_pred, _ = train_xgboost_gridsearch(X_train_fs, y_train, X_test_fs, y_test)
        details["hypertune"] = "Gridsearch"

    # Evaluation
    report, cm, acc = evaluate_model(y_test, y_pred)

    return {
        "Accuracy": acc,
        "Report": report,
        "Confusion": cm,
        "Model": model,
        "Class Labels": model.classes_,
        "Importances": importances_df,
        "Details": details,
    }

# Auto Best Mode: Try All Options
def run_auto_best() -> dict:
    """
    Tries all combinations of feature selection and hyperparameter tuning,
    returns the best-performing configuration.
    """
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target_col)
    results = []

    # Manual Features
    results.append(run_workflow({"fs_method": "Manual", "features": feature_cols, "hypertune": "No"}))
    results.append(run_workflow({"fs_method": "Manual", "features": feature_cols, "hypertune": "Gridsearch"}))

    # Random Forest Thresholds
    for thresh in [v for _, v in RF_THRESHOLDS]:
        for mode in XGB_MODES:
            results.append(run_workflow({"fs_method": "Random Forest", "rf_threshold": thresh, "hypertune": mode}))

    # L1 Alphas
    for alpha in L1_ALPHAS:
        for mode in XGB_MODES:
            results.append(run_workflow({"fs_method": "L1 (Lasso)", "l1_alpha": alpha, "hypertune": mode}))

    # Find Best Accuracy
    best_result = None
    best_acc = -1
    for res in results:
        if "Accuracy" in res and res["Accuracy"] > best_acc:
            best_acc = res["Accuracy"]
            best_result = res

    if best_result:
        best_result["AutoBest_Description"] = f"Auto (Best): {best_result['Details']}"
    return best_result if best_result else {"error": "Auto (Best) failed."}

# Main Execution
if st.button("Compare Workflows"):
    compare_results = []

    for i, cfg in enumerate(workflow_configs):
        with st.spinner(f"Running Workflow {i + 1}..."):
            if cfg["fs_method"] == "Auto (Best)":
                res = run_auto_best()
                if "error" in res:
                    st.error(f"Workflow {i + 1} error: {res['error']}")
                    st.stop()
                compare_results.append({
                    "Workflow": f"Workflow {i + 1}",
                    "Accuracy": res["Accuracy"],
                    "Report": res["Report"],
                    "Confusion": res["Confusion"],
                    "Model": res["Model"],
                    "Class Labels": res["Class Labels"],
                    "Importances": res["Importances"],
                    "AutoBest_Description": res.get("AutoBest_Description", None)
                })
            else:
                res = run_workflow(cfg)
                if "error" in res:
                    st.error(f"Workflow {i + 1} error: {res['error']}")
                    st.stop()
                compare_results.append({
                    "Workflow": f"Workflow {i + 1}",
                    "Accuracy": res["Accuracy"],
                    "Report": res["Report"],
                    "Confusion": res["Confusion"],
                    "Model": res["Model"],
                    "Class Labels": res["Class Labels"],
                    "Importances": res["Importances"],
                    "AutoBest_Description": None
                })

    # Display Results in Tabs
    tabs = st.tabs([f"Workflow {i + 1}" for i in range(len(compare_results))])

    for i, res in enumerate(compare_results):
        with tabs[i]:
            st.write(f"**{res['Workflow']}**")
            st.write(f"**Accuracy:** {res['Accuracy']:.3f}")
            if res["AutoBest_Description"]:
                st.info(res["AutoBest_Description"])
            render_classification_report(res["Report"])
            plot_confusion_matrix(res["Confusion"], res["Class Labels"])
            if res["Importances"] is not None:
                plot_importances(res["Importances"], title="Feature Importances")