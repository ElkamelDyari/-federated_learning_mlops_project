from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_best_model_params(experiment_name, metric="accuracy"):
    """
    Retrieve the hyperparameters of the best model from an MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        metric (str): The metric to sort runs by, in descending order.

    Returns:
        dict: A dictionary containing the model name and its parameters.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Query the runs sorted by the metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    best_run = runs[0]
    model_name = best_run.data.params.get("model_name")
    params = {k: v for k, v in best_run.data.params.items() if k != "model_name"}

    # Convert numeric parameters back to their original types
    for key, value in params.items():
        try:
            params[key] = eval(value)
        except (SyntaxError, NameError):
            params[key] = value  # Keep as string if eval fails

    return {"model_name": model_name, "params": params, "run_id": best_run.info.run_id}


def initialize_model(model_name, params):
    """
    Create a new model instance using the model name and parameters.

    Args:
        model_name (str): Name of the model (e.g., "SVM", "XGBoost").
        params (dict): Parameters for the model.

    Returns:
        Model: An instance of the model.
    """
    if model_name == "SVM":
        return SVC(**params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_name == "XGBoost":
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")



