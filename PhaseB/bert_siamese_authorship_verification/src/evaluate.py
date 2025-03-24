from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bert_siamese_authorship_verification.src.clustering import perform_kmedoids_clustering


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def evaluate_with_clustering(anomaly_scores):
    return perform_kmedoids_clustering(anomaly_scores, num_clusters=2)
