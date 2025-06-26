import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

def split_data(df, feature_cols, target_col, test_size=0.3, random_state=42):
    """
    Splits the dataset into train and test sets.
    """
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def rf_feature_selection(X_train, y_train, X_test, threshold="median"):
    """
    Feature selection using Random Forest and SelectFromModel.
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

def l1_feature_selection(X_train, y_train, X_test, alpha=0.01):
    """
    Feature selection using L1-penalized Logistic Regression (Lasso).
    """
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1/alpha, random_state=42, max_iter=2000)
    l1.fit(X_train, y_train)
    selector = SelectFromModel(l1, prefit=True)
    X_train_fs = selector.transform(X_train)
    X_test_fs = selector.transform(X_test)
    selected_features = list(X_train.columns[selector.get_support()])
    # Feature importances are abs(coefficients)
    importances = np.abs(l1.coef_).flatten()
    all_feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return X_train_fs, X_test_fs, selected_features, l1, all_feature_importances

def train_xgboost(X_train, y_train, X_test):
    """
    Trains XGBoost and predicts test data.
    """
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred

def train_xgboost_gridsearch(X_train, y_train, X_test, y_test):
    """
    Optimises and trains XGBoost with GridSearchCV.
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

def evaluate_model(y_test, y_pred):
    """
    Evaluates a model with classification report and confusion matrix.
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = report["accuracy"]
    return report, cm, accuracy