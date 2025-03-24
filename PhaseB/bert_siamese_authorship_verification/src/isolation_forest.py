from sklearn.ensemble import IsolationForest
from bert_siamese_authorship_verification.config.get_config import get_config

# Load config
config = get_config()


class AnomalyDetector:
    def __init__(self, number_of_trees=config['isolation_forest']['number_of_trees']):
        """
        Initializes the Isolation Forest model.

        :param number_of_trees: Number of estimators
        """
        self.model = IsolationForest(n_estimators=number_of_trees)

    def fit_predict(self, features):
        """
        Fits Isolation Forest on the data and predicts anomalies.

        :param features: List of feature vectors.
        :return: Boolean mask where True indicates normal samples.
        """
        predictions = self.model.fit_predict(features)
        return predictions == 1  # Keep only normal data points (1)
