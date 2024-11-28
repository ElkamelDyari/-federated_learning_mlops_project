import dagshub
import pandas as pd
import mlflow
import mlflow.xgboost
from src.data.feature_engineering import preprocess_data
from src.utilis.evaluate import evaluate_client  # Evaluate metrics
from src.utilis.best_model import get_best_model_params, initialize_model
from src.utilis.train_evaluate_client import train_and_evaluate_client


def load_client_data(paths, n_components=35, batch_size=500, test_size=0.3, random_state=0):
    return [preprocess_data(path, n_components=n_components, batch_size=batch_size, test_size=test_size,
                            random_state=random_state) for path in paths]


def federated_train(
        CLIENT_DATA_PATHS,
        EXPERIMENT_NAME="Federated Learning IDS",
        SEARCH_EXPERIMENT_NAME="grid_search"
):
    from mlflow.tracking import MlflowClient

    best_model_info = get_best_model_params(SEARCH_EXPERIMENT_NAME, metric="f1_score")
    print("Best Model Info:", best_model_info)

    model_name = best_model_info["model_name"]
    model_params = best_model_info["params"]

    global_model = initialize_model(model_name, model_params)
    print(f"Initialized {model_name} model with parameters: {model_params}")

    # Start MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="Federated_Learning_Cycle") as run:
        # Load and preprocess client data
        print("Loading and preprocessing client data...")
        client_data = load_client_data(CLIENT_DATA_PATHS)

        # Federated training across clients
        for i, (X_train, X_test, y_train, y_test) in enumerate(client_data, start=1):
            if i == 1:
                results = train_and_evaluate_client(global_model, X_train, X_test, y_train, y_test, client_id=i,
                                                    continues_training=False)
                print(f"Client {i} results: {results}")
            else:
                results = train_and_evaluate_client(global_model, X_train, X_test, y_train, y_test, client_id=i)
                print(f"Client {i} results: {results}")

        # Log the final global model to MLflow within the current experiment
        print("Saving final global model to current experiment...")
        logged_model_uri = mlflow.xgboost.log_model(global_model, artifact_path="federated_global_model")
        print(f"Model logged to: {logged_model_uri}")

    # Register the final model in the Model Registry
    MODEL_NAME = "FederatedGlobalModel"
    print(f"Registering final model to the Model Registry with name: {MODEL_NAME}...")
    client = MlflowClient()
    try:
        registered_model = client.create_registered_model(MODEL_NAME)
        print(f"Created a new registered model: {MODEL_NAME}")
    except mlflow.exceptions.MlflowException:
        print(f"Model {MODEL_NAME} already exists in the registry.")

    # Register the model version
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=logged_model_uri.model_uri,
        run_id=run.info.run_id
    )
    print(f"Registered model version: {model_version.version}")
    print(f"Model {MODEL_NAME} registered successfully in the Model Registry.")


def main():
    # Paths to client data
    CLIENT_DATA_PATHS = [
        "data/processed/data1.csv",
        "data/processed/data2.csv",
        "data/processed/data3.csv",
        "data/processed/data4.csv"
    ]

    mlflow.set_tracking_uri("https://dagshub.com/ElkamelDyari/-federated_learning_mlops_project.mlflow")
    dagshub.init(repo_owner='ElkamelDyari', repo_name='-federated_learning_mlops_project', mlflow=True)
    federated_train(CLIENT_DATA_PATHS)


if __name__ == "__main__":
    main()
