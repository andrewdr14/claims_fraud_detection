import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from claims_fraud.model import (
    split_data,
    feature_selection,
    train_xgboost,
    train_xgboost_gridsearch,
    train_catboost,
    train_catboost_gridsearch,
    evaluate_model,
    handle_imbalance
)

# Load dataset
df = pd.read_csv("claims_fraud/cleaned_insurance_claims.csv")
target_col = "fraud_reported"
feature_cols = [col for col in df.columns if col != target_col]

# Constants
MODEL_OPTIONS = ["XGBoost", "CatBoost"]
FS_METHODS = ["Manual", "Random Forest", "L1 (Lasso)"] # Removed "Auto (Best)"
XGB_MODES = ["No", "Gridsearch"]
IMBALANCE_HANDLING_METHODS = ["None", "Random Oversampler", "SMOTE", "Class Weighting"]

RF_THRESHOLDS = [
    ("Median (default)", "median"),
    ("Mean", "mean"),
    ("Top 25% (0.75*mean)", "0.75*mean"),
    ("Top 50% (0.5*mean)", "0.5*mean"),
    ("Top 75% (0.25*mean)", "0.25*mean"),
]
L1_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
MAX_WORKFLOWS = 4

# Initialize session state
st.session_state.setdefault("workflow_ids", [1])
st.session_state.setdefault("results", [])

def next_wf_id() -> int:
    if "next_wf_id" not in st.session_state:
        st.session_state["next_wf_id"] = 2
    val = st.session_state["next_wf_id"]
    st.session_state["next_wf_id"] += 1
    return val

# --- UI Functions ---
def render_classification_report(report_dict: dict, title: str = "Classification Report"):
    st.subheader(title)
    report_df = pd.DataFrame(report_dict).T
    st.dataframe(report_df.style.format(precision=2))

def plot_confusion_matrix(cm: np.ndarray, class_labels: list, title: str = "Confusion Matrix"):
    st.subheader(title)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def plot_importances(importances_df: pd.DataFrame, title: str = "Feature Importances"):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, min(0.5 * len(importances_df), 8)))
    sns.barplot(data=importances_df, x="importance", y="feature", ax=ax, color="royalblue")
    ax.set_title(title)
    st.pyplot(fig)

def get_workflow_config(workflow_num: int, workflow_id: int, allow_remove: bool) -> dict:
    # Use a container for the workflow to apply border and consistent styling
    with st.container(border=True):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(f"#### Workflow {workflow_num + 1}")
        with col2:
            # Only show remove button if more than one workflow exists
            if len(st.session_state["workflow_ids"]) > 1:
                if st.button("✖️", key=f"remove_wf_{workflow_id}", help="Remove this workflow"):
                    st.session_state["workflow_ids"].remove(workflow_id)
                    st.rerun()

        model_choice = st.selectbox("Training Model", MODEL_OPTIONS, key=f"model_{workflow_id}")
        fs_method = st.selectbox("Feature Selection Method", FS_METHODS, key=f"fs_{workflow_id}")

        config = {"model": model_choice, "fs_method": fs_method, "remove_clicked": False, "workflow_id": workflow_id}

        if fs_method == "Manual":
            config["features"] = st.multiselect("Select Features", feature_cols, default=feature_cols, key=f"mf_{workflow_id}")
        elif fs_method == "Random Forest":
            rf_label_to_value = {label: value for label, value in RF_THRESHOLDS}
            labels = [label for label, _ in RF_THRESHOLDS]
            selected = st.selectbox("Importance Threshold", labels, index=0, key=f"rfthresh_{workflow_id}")
            config["rf_threshold"] = rf_label_to_value[selected]
        elif fs_method == "L1 (Lasso)":
            alpha = st.number_input("L1 Regularization Strength (Alpha)", 0.0001, 1.0, 0.01, 0.01, key=f"l1alpha_{workflow_id}")
            config["l1_alpha"] = alpha

        # Hyperparameter Tuning
        mode = st.selectbox("Hyperparameter Tuning", XGB_MODES, key=f"ht_{workflow_id}")
        config["hypertune"] = mode

        # Imbalance Handling
        imbalance_method = st.selectbox("Imbalance Handling", IMBALANCE_HANDLING_METHODS, key=f"imbalance_{workflow_id}")
        config["imbalance_method"] = imbalance_method

        # Add random state input
        random_state = st.number_input("Random State", value=42, step=1, key=f"random_state_{workflow_id}")
        config["random_state"] = random_state

    return config

# --- Core Logic ---
def run_workflow(cfg: dict) -> dict:
    random_state = cfg.get("random_state", 42) # Get random_state from config, default to 42
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target_col, random_state=random_state)
    details = {}
    importances_df = None

    # Feature Selection
    if cfg["fs_method"] == "Manual":
        if not cfg.get("features"): return {"error": "No features selected."}
        X_train_fs, X_test_fs = X_train[cfg["features"]], X_test[cfg["features"]]
        details["features"] = cfg["features"]
    elif cfg["fs_method"] in ["Random Forest", "L1 (Lasso)"]:
        method = "Random Forest" if cfg["fs_method"] == "Random Forest" else "L1"
        X_train_fs, X_test_fs, _, _, importances_df = feature_selection(
            method, X_train, y_train, X_test,
            threshold=cfg.get("rf_threshold"), alpha=cfg.get("l1_alpha"), random_state=random_state
        )
        details.update({k: v for k, v in cfg.items() if k in ["rf_threshold", "l1_alpha"]})
    else:
        X_train_fs, X_test_fs = X_train, X_test

    # Imbalance Handling
    X_train_processed, y_train_processed = X_train_fs.copy(), y_train.copy()
    scale_pos_weight = 1.0  # Reset to default
    class_weights = None    # Reset to default

    if cfg["imbalance_method"] == "Random Oversampler":
        X_train_processed, y_train_processed = handle_imbalance(X_train_fs, y_train, "RandomOverSampler", random_state=random_state)
    elif cfg["imbalance_method"] == "SMOTE":
        X_train_processed, y_train_processed = handle_imbalance(X_train_fs, y_train, "SMOTE", random_state=random_state)
    elif cfg["imbalance_method"] == "Class Weighting":
        # Calculate scale_pos_weight for XGBoost
        neg_count = y_train.value_counts()[0] # Assuming 0 is the majority class
        pos_count = y_train.value_counts()[1] # Assuming 1 is the minority class
        scale_pos_weight = neg_count / pos_count
        
        # Calculate class_weights for CatBoost
        class_weights = {0: 1, 1: scale_pos_weight} # Assuming 0 is majority, 1 is minority

    # Train Model
    if cfg.get("hypertune") == "No":
        if cfg["model"] == "XGBoost": 
            model, y_pred = train_xgboost(X_train_processed, y_train_processed, X_test_fs, scale_pos_weight=scale_pos_weight, random_state=random_state)
        else: 
            model, y_pred = train_catboost(X_train_processed, y_train_processed, X_test_fs, class_weights=class_weights, random_state=random_state)
        details["hypertune"] = "No"
    else:
        if cfg["model"] == "XGBoost": 
            model, y_pred, _ = train_xgboost_gridsearch(X_train_processed, y_train_processed, X_test_fs, y_test, scale_pos_weight=scale_pos_weight, random_state=random_state)
        else: 
            model, y_pred, _ = train_catboost_gridsearch(X_train_processed, y_train_processed, X_test_fs, y_test, class_weights=class_weights, random_state=random_state)
        details["hypertune"] = "Gridsearch"

    report, cm, acc, class_labels = evaluate_model(y_test, y_pred)
    return {"Accuracy": acc, "Report": report, "Confusion": cm, "Model": model, "Class Labels": class_labels, "Importances": importances_df, "Details": details, "Config": cfg} # Store the config

# --- Main App ---
st.title("Compare Machine Learning Workflows for Fraud Detection")

st.markdown("""
This page allows you to set up and compare up to 4 custom machine learning workflows side-by-side. Experiment with different models, feature selection methods, hyperparameter tuning strategies, and imbalance handling techniques to see how they impact model performance.

For detailed explanations of the concepts and techniques used here, please visit the [Concepts page](Concepts).

### How to Use This Page:
1.  **Configure Workflows:** Use the dropdowns and options in each workflow column to define your desired settings. You can add more workflows (up to 4) using the "➕ Add Workflow" button.
2.  **Run Comparison:** Click the "Compare Workflows" button to train and evaluate all configured models.
3.  **Analyze Results:** The results section below will display a summary table and detailed metrics (Classification Report, Confusion Matrix, Feature Importances) for each workflow. Use the dropdowns in the results tabs to cross-compare different metrics across your workflows.
""")

# --- Workflow Configuration ---

cols = st.columns(len(st.session_state["workflow_ids"]))
workflow_configs = []
remove_indices = []

for i, workflow_id in enumerate(st.session_state["workflow_ids"]):
    with cols[i]:
        # Removed div for custom border
        cfg = get_workflow_config(i, workflow_id, True) # Always allow remove button within the config function
        workflow_configs.append(cfg)
        if cfg["remove_clicked"]:
            remove_indices.append(i)

if remove_indices:
    for idx in sorted(remove_indices, reverse=True):
        del st.session_state["workflow_ids"][idx]
    st.rerun()

# --- Add/Compare Buttons ---
# Adjust column layout for centering
if len(st.session_state["workflow_ids"]) < MAX_WORKFLOWS:
    c1, c2 = st.columns([0.5, 0.5]) # Two columns when Add Workflow is present
    with c1:
        if st.button("➕ Add Workflow", use_container_width=True):
            st.session_state["workflow_ids"].append(next_wf_id())
            st.rerun()
    with c2:
        if st.button("Compare Workflows", use_container_width=True, type="primary"):
            results = []
            for i, cfg in enumerate(workflow_configs):
                with st.spinner(f"Running Workflow {i + 1}..."):
                    res = run_workflow(cfg)
                    if "error" in res:
                        st.error(f"Workflow {i + 1} error: {res['error']}")
                        continue
                    results.append(res)
            st.session_state["results"] = results
else:
    # Center the Compare Workflows button when Add Workflow is not present
    c1, c2, c3 = st.columns([0.3, 0.4, 0.3]) # Three columns, middle one for button
    with c2:
        if st.button("Compare Workflows", use_container_width=True, type="primary"):
            results = []
            for i, cfg in enumerate(workflow_configs):
                with st.spinner(f"Running Workflow {i + 1}..."):
                    res = run_workflow(cfg)
                    if "error" in res:
                        st.error(f"Workflow {i + 1} error: {res['error']}")
                        continue
                    results.append(res)
            st.session_state["results"] = results

# --- Results Display ---
if st.session_state["results"]:
    st.markdown("---")
    st.header("Results")

    # Summary Tab
    summary_data = {}
    for i, res in enumerate(st.session_state["results"]):
        # Use the stored config from the results, not the current workflow_configs
        cfg_stored = res["Config"]
        fs_details = cfg_stored.get("fs_method", "N/A")
        if cfg_stored.get("fs_method") == "Random Forest":
            fs_details += f" (Thresh: {cfg_stored.get('rf_threshold')})"
        elif cfg_stored.get("fs_method") == "L1 (Lasso)":
            fs_details += f" (Alpha: {cfg_stored.get('l1_alpha')})"

        summary_data[f"Workflow {i+1}"] = {
            "Model": cfg_stored.get("model", "N/A"), # Use stored model choice
            "Accuracy": f"{res.get('Accuracy', 0):.3f}",
            "Feature Selection": fs_details,
            "Hyperparameter Tuning": cfg_stored.get("hypertune", "N/A"),
            "Imbalance Handling": cfg_stored.get("imbalance_method", "N/A")
        }

    summary_df = pd.DataFrame(summary_data).T
    
    main_tab_titles = ["Summary", "Classification Report", "Confusion Matrix", "Feature Importances"]
    main_tabs = st.tabs(main_tab_titles)

    with main_tabs[0]: # Summary Tab
        st.subheader("Comparison Summary")
        st.dataframe(summary_df.style.highlight_max(axis=0, subset=['Accuracy'], color='lightgreen'))

    # Create a list of workflow options for selectboxes
    workflow_options = [f"Workflow {i+1}" for i in range(len(st.session_state["results"]))]

    with main_tabs[1]: # Classification Report Tab
        st.subheader("Classification Report")
        selected_wf_report = st.selectbox("Select Workflow for Report", workflow_options, key="select_report_wf")
        selected_index_report = workflow_options.index(selected_wf_report)
        res_report = st.session_state["results"][selected_index_report]
        render_classification_report(res_report["Report"], title=f"Classification Report for {selected_wf_report}")

    with main_tabs[2]: # Confusion Matrix Tab
        st.subheader("Confusion Matrix")
        selected_wf_cm = st.selectbox("Select Workflow for Confusion Matrix", workflow_options, key="select_cm_wf")
        selected_index_cm = workflow_options.index(selected_wf_cm)
        res_cm = st.session_state["results"][selected_index_cm]
        plot_confusion_matrix(res_cm["Confusion"], res_cm["Class Labels"], title=f"Confusion Matrix for {selected_wf_cm}")

    with main_tabs[3]: # Feature Importances Tab
        st.subheader("Feature Importances")
        selected_wf_fi = st.selectbox("Select Workflow for Feature Importances", workflow_options, key="select_fi_wf")
        selected_index_fi = workflow_options.index(selected_wf_fi)
        res_fi = st.session_state["results"][selected_index_fi]
        if res_fi["Importances"] is not None:
            plot_importances(res_fi["Importances"], title=f"Feature Importances for {selected_wf_fi}")
        else:
            st.info("Feature importances are not available for this workflow (e.g., Manual feature selection). ")
