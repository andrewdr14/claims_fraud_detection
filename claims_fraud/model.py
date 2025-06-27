"""
Model utilities for fraud detection including feature selection, training, and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full input dataframe containing features and target column.
    feature_cols : list of str
        List of feature column names to use for model training.
    target_col : str
        Name of the target column.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Controls the shuffling applied before the split.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Test feature set.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    """
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def rf_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = "median"
) -> tuple[np.ndarray, np.ndarray, list[str], RandomForestClassifier, pd.DataFrame]:
    """
    Perform feature selection using Random Forest importance scores.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test feature set.
    threshold : str, default="median"
        Threshold used to select features based on importance scores.

    Returns
    -------
    X_train_fs : np.ndarray
        Transformed training feature set after selection.
    X_test_fs : np.ndarray
        Transformed test feature set after selection.
    selected_features : list of str
        List of selected feature names.
    rf : RandomForestClassifier
        Fitted Random Forest model.
    all_feature_importances : pd.DataFrame
        DataFrame with all features and their importance scores.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    selector = SelectFromModel(rf, prefit=True, threshold=threshold)
    X_train_fs = selector.transform(X_train)
    X_test_fs = selector.transform(X_test)

    selected_features = list(X_train.columns[selector.get_support()])
    all_feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    return X_train_fs, X_test_fs, selected_features, rf, all_feature_importances


def l1_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.01
) -> tuple[np.ndarray, np.ndarray, list[str], LogisticRegression, pd.DataFrame]:
    """
    Perform feature selection using L1-penalized logistic regression (Lasso).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test feature set.
    alpha : float, default=0.01
        Regularization strength; larger values specify stronger regularization.

    Returns
    -------
    X_train_fs : np.ndarray
        Transformed training feature set after selection.
    X_test_fs : np.ndarray
        Transformed test feature set after selection.
    selected_features : list of str
        List of selected feature names.
    l1 : LogisticRegression
        Fitted L1-penalized logistic regression model.
    all_feature_importances : pd.DataFrame
        DataFrame with all features and their absolute coefficient magnitudes.
    """
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1/alpha, random_state=42, max_iter=2000)
    l1.fit(X_train, y_train)

    selector = SelectFromModel(l1, prefit=True)
    X_train_fs = selector.transform(X_train)
    X_test_fs = selector.transform(X_test)

    selected_features = list(X_train.columns[selector.get_support()])
    importances = np.abs(l1.coef_).flatten()

    all_feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return X_train_fs, X_test_fs, selected_features, l1, all_feature_importances


def train_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray
) -> tuple[XGBClassifier, np.ndarray]:
    """
    Train an XGBoost classifier and make predictions on the test set.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature set.
    y_train : pd.Series
        Training labels.
    X_test : np.ndarray
        Test feature set.

    Returns
    -------
    xgb : XGBClassifier
        Trained XGBoost model.
    y_pred : np.ndarray
        Predicted labels for the test set.
    """
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred


def train_xgboost_gridsearch(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> tuple[XGBClassifier, np.ndarray, dict]:
    """
    Optimize and train XGBoost using grid search over hyperparameters.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature set.
    y_train : pd.Series
        Training labels.
    X_test : np.ndarray
        Test feature set.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    best_xgb : XGBClassifier
        Best trained XGBoost model found via grid search.
    y_pred : np.ndarray
        Predicted labels for the test set.
    best_params : dict
        Dictionary of best hyperparameter values found during grid search.
    """
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


def evaluate_model(
    y_test: pd.Series,
    y_pred: np.ndarray
) -> tuple[dict, np.ndarray, float]:
    """
    Evaluate model performance using classification report and confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
        True test labels.
    y_pred : np.ndarray
        Predicted labels from the model.

    Returns
    -------
    report : dict
        Classification report dictionary.
    cm : np.ndarray
        Confusion matrix array.
    accuracy : float
        Accuracy score computed from the classification report.
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = report["accuracy"]
    return report, cm, accuracy