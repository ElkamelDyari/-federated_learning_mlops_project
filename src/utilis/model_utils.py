import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utilis.evaluate import evaluate_model


def initialize_model(model_name, model_params):
    if model_name == "SVM":
        return SVC(**model_params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**model_params)
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(**model_params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**model_params)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**model_params)
    elif model_name == "XGBoost":
        return XGBClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_client_data(paths, n_components=35, batch_size=500, test_size=0.3, random_state=0):
    from src.data.feature_engineering import preprocess_data
    return [preprocess_data(path, n_components=n_components, batch_size=batch_size, test_size=test_size, random_state=random_state) for path in paths]

def train_and_evaluate_client(model, X_train, X_test, y_train, y_test, client_id, continues_training=True):
    if continues_training:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    predictions = model.predict(X_test)
    return evaluate_model(y_test, predictions)