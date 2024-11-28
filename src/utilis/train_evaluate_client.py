from sklearn.metrics import confusion_matrix

from src.utilis.evaluate import evaluate_client


def train_and_evaluate_client(global_model, X_train, X_test, y_train, y_test, client_id, continues_training=True):
    """
    Train the global model on a client's data and evaluate its performance.

    Args:
        global_model: The current global model to be trained and evaluated.
        X_train: Training feature data for the client.
        X_test: Testing feature data for the client.
        y_train: Training target labels for the client.
        y_test: Testing target labels for the client.
        client_id: An integer identifier for the client.
        continues_training: if True fit the model on the previous weights for continues training
    Returns:
        dict: A dictionary containing training and testing metrics for the client.
    """
    import mlflow

    print(f"\nTraining on client {client_id}...")

    if continues_training :
        # Continue training the global model on the client's data
        global_model.fit(X_train, y_train, xgb_model=global_model)
    else :
        global_model.fit(X_train, y_train)

    # Evaluate model on client data
    print(f"Evaluating on client {client_id} data...")
    train_metrics = evaluate_client(global_model, X_train, y_train, data_type="Train", client_id=client_id)
    test_metrics = evaluate_client(global_model, X_test, y_test, data_type="Test", client_id=client_id)

    print(f"client {client_id} train: {train_metrics}, \ntest: {test_metrics}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        f"client_{client_id}_train_accuracy": train_metrics['Accuracy'],
        f"client_{client_id}_train_precision": train_metrics['Precision'],
        f"client_{client_id}_train_recall": train_metrics['Recall'],
        f"client_{client_id}_train_f1_score": train_metrics['F1_Score'],
        f"client_{client_id}_test_accuracy": test_metrics['Accuracy'],
        f"client_{client_id}_test_precision": test_metrics['Precision'],
        f"client_{client_id}_test_recall": test_metrics['Recall'],
        f"client_{client_id}_test_f1_score": test_metrics['F1_Score'],
    })

    # Return the evaluation metrics
    return {"train_metrics": train_metrics, "test_metrics": test_metrics}
