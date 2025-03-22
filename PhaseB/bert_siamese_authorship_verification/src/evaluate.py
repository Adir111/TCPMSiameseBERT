from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(y_true, y_pred):
    """
    Evaluates model performance using accuracy, F1-score, precision, and recall.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Dictionary of evaluation metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
