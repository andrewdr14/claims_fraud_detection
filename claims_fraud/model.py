"""
Model utilities for fraud detection including feature selection, training, and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Import resampling techniques
from imblearn.over_sampling import RandomOverSampler, SMOTE


# Type alias for supported models
SupportedModel = Union[XGBClassifier, CatBoostClassifier]


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into train and test sets.

    Parameters:
        df (pd.DataFrame): The full input dataframe containing features and target column.
        feature_cols (list[str]): List of feature column names to use for model training.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied before the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def handle_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies resampling techniques to handle class imbalance.

    Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        method (str): Resampling method ('RandomOverSampler', 'SMOTE').
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled X and y.
    """
    if method == "RandomOverSampler":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "SMOTE":
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unsupported imbalance handling method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


def feature_selection(
    method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = "median",
    alpha: float = 0.01,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, list[str], Any, pd.DataFrame]:
    """
    General-purpose feature selection using specified method.

    Supported methods:
        - Random Forest: Uses importance scores from a trained random forest.
        - L1: Uses L1-penalized logistic regression to zero out less important features.

    Parameters:
        method (str): Feature selection method ('Random Forest' or 'L1').
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test feature set.
        threshold (str): Threshold used in Random Forest selector.
        alpha (float): Regularization strength in L1 selector.

    Returns:
        Tuple[np.ndarray, np.ndarray, list[str], Any, pd.DataFrame]:
            X_train_fs, X_test_fs, selected_features, model_used, importances_df
    """
    if method == "Random Forest":
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)

        selector = SelectFromModel(rf, prefit=True, threshold=threshold)
        X_train_fs = selector.transform(X_train)
        X_test_fs = selector.transform(X_test)

        selected_features = list(X_train.columns[selector.get_support()])
        importances_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": rf.feature_importances_
        }).sort_values(by="importance", ascending=False)

        return X_train_fs, X_test_fs, selected_features, rf, importances_df

    elif method == "L1":
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1 / alpha, max_iter=2000, random_state=random_state)
        l1.fit(X_train, y_train)

        selector = SelectFromModel(l1, prefit=True)
        X_train_fs = selector.transform(X_train)
        X_test_fs = selector.transform(X_test)

        selected_features = list(X_train.columns[selector.get_support()])
        importances = np.abs(l1.coef_).flatten()

        importances_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        return X_train_fs, X_test_fs, selected_features, l1, importances_df

    else:
        raise ValueError(f"Unsupported feature selection method: {method}")


def train_catboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    class_weights: Dict[int, float] = None,
    random_state: int = 42
) -> Tuple[CatBoostClassifier, np.ndarray]:
    """
    Train a CatBoost classifier and make predictions on the test set.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Test feature set.
        class_weights (Dict[int, float]): Dictionary of class weights.

    Returns:
        Tuple[CatBoostClassifier, np.ndarray]: Trained model and predicted labels
    """
    cb = CatBoostClassifier(verbose=False, random_state=random_state, class_weights=class_weights)
    cb.fit(X_train, y_train)
    y_pred = cb.predict(X_test)
    return cb, y_pred


def train_catboost_gridsearch(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    class_weights: Dict[int, float] = None,
    random_state: int = 42
) -> Tuple[CatBoostClassifier, np.ndarray, dict]:
    """
    Optimize and train CatBoost using grid search over hyperparameters.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (pd.Series): Test labels.
        class_weights (Dict[int, float]): Dictionary of class weights.

    Returns:
        Tuple[CatBoostClassifier, np.ndarray, dict]: Best model, predictions, best parameters
    """
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200],
        'l2_leaf_reg': [1, 3, 5]
    }

    cb = CatBoostClassifier(random_state=random_state, verbose=False, class_weights=class_weights)
    grid_search = GridSearchCV(estimator=cb, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=3, verbose=0)
    grid_search.fit(X_train, y_train)

    best_cb = grid_search.best_estimator_
    y_pred = best_cb.predict(X_test)

    return best_cb, y_pred, grid_search.best_params_


def train_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_state: int = 42
) -> Tuple[XGBClassifier, np.ndarray]:
    """
    Train an XGBoost classifier and make predictions on the test set.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Test feature set.
        scale_pos_weight (float): Controls the balance of positive and negative weights.

    Returns:
        Tuple[XGBClassifier, np.ndarray]: Trained model and predicted labels
    """
    xgb = XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1, scale_pos_weight=scale_pos_weight)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred


def train_xgboost_gridsearch(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    scale_pos_weight: float = 1.0,
    random_state: int = 42
) -> Tuple[XGBClassifier, np.ndarray, dict]:
    """
    Optimize and train XGBoost using grid search over hyperparameters.

    Parameters:
        X_train (np.ndarray): Training feature set.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (pd.Series): Test labels.
        scale_pos_weight (float): Controls the balance of positive and negative weights.

    Returns:
        Tuple[XGBClassifier, np.ndarray, dict]: Best model, predictions, best parameters
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1, scale_pos_weight=scale_pos_weight)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=3, verbose=0)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)

    return best_xgb, y_pred, grid_search.best_params_


def evaluate_model(
    y_test: pd.Series,
    y_pred: np.ndarray
) -> Tuple[Dict, np.ndarray, float, list]:
    """
    Evaluate model performance using classification report and confusion matrix.

    Parameters:
        y_test (pd.Series): True test labels.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns:
        Tuple[Dict, np.ndarray, float, list]: Report dictionary, confusion matrix, accuracy, class labels
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = report["accuracy"]
    class_labels = sorted(list(y_test.unique())) # Get unique class labels from y_test
    return report, cm, accuracy, class_labels