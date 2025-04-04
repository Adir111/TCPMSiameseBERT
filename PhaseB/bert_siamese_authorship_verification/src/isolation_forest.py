from sklearn.ensemble import IsolationForest
from config.get_config import get_config

# Load config
config = get_config()


class AnomalyDetector:
    def __init__(self, number_of_trees=config['isolation_forest']['number_of_trees']):
        """
        Initializes the Isolation Forest model.

        :param number_of_trees: Number of estimators
        """
        self.model = IsolationForest(n_estimators=number_of_trees)

    def fit_score(self, features):
        """
        Fits Isolation Forest on the data and returns anomaly scores.

        :param features: List of feature vectors.
        :return: Anomaly score vector (higher = more anomalous).
        """
        self.model.fit(features)
        scores = self.model.decision_function(features)  # higher = more normal
        return scores

    # Used to test the pipeline
    def fit_predict(self, features):
        """
        Classifies each sample as normal (1) or anomaly (0).
        :param features: numpy array or list of feature vectors
        :return: Array of 0 (anomaly) or 1 (normal)
        """
        self.model.fit(features)
        return (self.model.predict(features) == 1).astype(int)
