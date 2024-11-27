from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc


def evaluate_client(model, X, y, data_type="Test", client_id=0):
    """
    Evaluate the model on the given dataset.
    :param model: Trained model
    :param X: Features
    :param y: Labels
    :param data_type: 'Train' or 'Test'
    :param client_id: Client identifier
    :return: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X)
    metrics = {
        'Client': client_id,
        'Data': data_type,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1_Score': f1_score(y, y_pred, average='weighted')
    }

    return metrics


def evaluate_model(y_true, predictions, proba_predictions=None):
    """
    Evaluates a binary classification model and computes various metrics.

    Parameters:
    - y_true: array-like, true labels.
    - predictions: array-like, predicted labels.
    - proba_predictions: array-like, predicted probabilities for the positive class (optional).

    Returns:
    - metrics: dict containing calculated metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "f1_score": f1_score(y_true, predictions),
        "f2_score": fbeta_score(y_true, predictions, beta=2),
        "f0.5_score": fbeta_score(y_true, predictions, beta=0.5),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "auc_roc": roc_auc_score(y_true, proba_predictions) if proba_predictions is not None else 0,
    }

    # Precision-Recall AUC
    if proba_predictions is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba_predictions)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    else:
        metrics["pr_auc"] = 0

    return metrics
