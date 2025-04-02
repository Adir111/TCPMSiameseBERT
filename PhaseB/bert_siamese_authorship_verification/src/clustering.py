import numpy as np
from sklearn_extra.cluster import KMedoids


def perform_kmedoids_clustering(anomaly_scores, num_clusters=2):
    """
    Clusters anomaly scores using K-Medoids.
    
    :param anomaly_scores: List of anomaly scores from Isolation Forest.
    :param num_clusters: Number of clusters (default=2: same author, different author).
    :return: Cluster labels
    """
    if len(anomaly_scores) < 2:
        print("[ERROR] Not enough tested texts for clustering. Need at least 2.")
        return

    anomaly_scores = np.array(anomaly_scores).reshape(-1, 1)
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
    return kmedoids.fit_predict(anomaly_scores)
