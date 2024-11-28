import mlflow
import dagshub
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.utilis.evaluate import evaluate_model
from src.data.feature_engineering import preprocess_data
import mlflow.sklearn
import mlflow.xgboost
import json

mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/-federated_learning_mlops_project.mlflow")
dagshub.init(repo_owner='ElkamelDyari', repo_name='-federated_learning_mlops_project', mlflow=True)


def grid_Search(path, grid_params, experiment_name):
    mlflow.set_experiment(experiment_name)

    # Load and split your dataset
    X_train, X_test, y_train, y_test = preprocess_data(file_path=path)

    # Perform grid search and log results
    for model_name, param_grid in grid_params.items():
        for params in ParameterGrid(param_grid):
            with mlflow.start_run(run_name=f"{model_name}{params}_grid_search"):
                # Initialize the model
                if model_name == "SVM":
                    model = SVC(**params)
                elif model_name == "KNN":
                    model = KNeighborsClassifier(**params)
                elif model_name == "DecisionTree":
                    model = DecisionTreeClassifier(**params)
                elif model_name == "RandomForest":
                    model = RandomForestClassifier(**params)
                elif model_name == "LogisticRegression":
                    model = LogisticRegression(**params)
                elif model_name == "XGBoost":
                    model = XGBClassifier(**params)

                # Train and evaluate
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                proba_predictions = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                metrics = evaluate_model(y_test, predictions, proba_predictions)

                # Log metrics in MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(params)

                # Log the model using the appropriate MLflow flavor
                if model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")


def main():
    # Set up your parameter grids
    # param_grids = {
    #    "SVM": {"kernel": ["linear", "rbf"],
    #            "C": [0.1, 1],
    #            'probability': [True]},
    #    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    #    "DecisionTree": {"criterion": ["gini", "entropy"],
    #                     "max_depth": [5, 8, 10, 20, None]},
    #    "RandomForest": {"n_estimators": [10, 20, 30],
    #                     "max_depth": [5, 8, 10, 15]},
    #    "LogisticRegression": {'max_iter': [100, 500, 1000, 5000, 10000],
    #                           "C": [0.1, 1, 10],
    #                           'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg']},
    #    "XGBoost": {'n_estimators': [10, 20, 50],
    #                'max_depth': [3, 5, 10],
    #                'learning_rate': [0.01, 0.1],
    #                'subsample': [0.6, 1.0],
    #                'colsample_bytree': [0.6, 1.0]},
    # }

    # Set up your parameter grids
    param_grids = {
        "XGBoost": {'n_estimators': [50],
                    'max_depth': [10],
                    'learning_rate': [0.01],
                    'subsample': [0.6],
                    'colsample_bytree': [1.0]},
    }

    file_path = "data/processed/data1.csv"

    try:
        grid_Search(file_path, param_grids, "Grid_Search")
    except Exception as e:
        raise Exception(f"An error occurred :{e}")


if __name__ == "__main__":
    main()
