import streamlit as st

st.title("Machine Learning Concepts")

st.markdown("""
This page provides explanations of the key machine learning concepts and techniques used in this application.
""")

st.header("Machine Learning Models")

st.subheader("XGBoost (Extreme Gradient Boosting)")
st.markdown("""
XGBoost is a highly efficient, flexible, and portable open-source library that implements gradient boosting machines. It's known for its speed and performance, especially on structured or tabular data. XGBoost builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. It includes regularization techniques to prevent overfitting, making it a robust choice for many classification and regression tasks.
""")

st.subheader("CatBoost")
st.markdown("""
CatBoost is another powerful open-source gradient boosting library developed by Yandex. Its key advantage is its native handling of categorical features, which often require extensive preprocessing (like one-hot encoding) in other algorithms. CatBoost uses a permutation-driven approach to handle categorical features, reducing the need for manual feature engineering and often leading to better performance and faster training times.
""")

st.header("Feature Selection Methods")

st.subheader("Manual Feature Selection")
st.markdown("""
In manual feature selection, you explicitly choose which features from your dataset to include in the model training. This method relies on domain knowledge, exploratory data analysis (EDA), or prior experience to identify relevant features. While it offers full control, it can be time-consuming and may miss important interactions between features.
""")

st.subheader("Random Forest Feature Importance")
st.markdown("""
Random Forest is an ensemble learning method that builds multiple decision trees. During the training process, Random Forest can calculate the importance of each feature based on how much it contributes to reducing impurity (e.g., Gini impurity) across all trees in the forest. Features with higher importance scores are considered more relevant for prediction. This method provides a data-driven way to select features, helping to reduce dimensionality and improve model performance.
""")

st.subheader("L1 (Lasso) Regularization")
st.markdown("""
L1 Regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator), is a technique used in linear models (like Logistic Regression) to prevent overfitting and perform feature selection. It adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function. This penalty encourages some feature coefficients to become exactly zero, effectively removing those features from the model. Higher regularization strength (alpha) leads to more features being eliminated.
""")

st.header("Hyperparameter Tuning")

st.subheader("Grid Search")
st.markdown("""
Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model. Hyperparameters are parameters whose values are set before the learning process begins (e.g., learning rate, tree depth). Grid Search is a common method for hyperparameter tuning. It exhaustively searches over a specified subset of the hyperparameter space, trying every possible combination of values. For each combination, the model is trained and evaluated (typically using cross-validation), and the combination that yields the best performance is selected.
""")

st.header("Imbalance Handling Techniques")

st.markdown("""
Class imbalance occurs when the number of observations belonging to one class is significantly lower than those belonging to other classes. In fraud detection, fraudulent cases (minority class) are typically much rarer than non-fraudulent cases (majority class). This imbalance can lead to models that perform poorly on the minority class, as they tend to optimize for overall accuracy by predicting the majority class more often.
""")

st.subheader("Random Oversampler")
st.markdown("""
Random Oversampling is a simple technique to address class imbalance. It involves randomly duplicating instances from the minority class until the class distribution is more balanced. While straightforward, it can lead to overfitting if the duplicated samples are too similar, and it doesn't add any new information to the dataset.
""")

st.subheader("SMOTE (Synthetic Minority Over-sampling Technique)")
st.markdown("""
SMOTE is a more sophisticated oversampling technique. Instead of simply duplicating existing minority class samples, SMOTE generates synthetic samples. It works by selecting a minority class instance and finding its k-nearest neighbors. New synthetic instances are then created along the line segments joining the minority instance and its randomly selected neighbors. This approach helps to create a more diverse set of synthetic samples, reducing the risk of overfitting compared to random oversampling.
""")

st.subheader("Class Weighting")
st.markdown("""
Class weighting is a technique where different weights are assigned to the classes during the model training process. For imbalanced datasets, a higher weight is assigned to the minority class and a lower weight to the majority class. This tells the model to pay more attention to correctly classifying instances from the minority class, effectively penalizing misclassifications of the minority class more heavily. Many machine learning algorithms (like XGBoost and CatBoost) have built-in parameters (e.g., `scale_pos_weight` for XGBoost, `class_weights` for CatBoost) to implement this.
""")