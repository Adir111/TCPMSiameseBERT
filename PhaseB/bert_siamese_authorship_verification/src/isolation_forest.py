import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

from .data_loader import DataLoader

class DTWIsolationForest:
    def __init__(self, config, logger):
        self.logger = logger
        self.n_estimators = int(config["isolation_forest"]['number_of_trees'])
        self.percentile_threshold = float(config["isolation_forest"]['percentile_threshold'])
        self.anomaly_score_threshold = float(config["isolation_forest"]['anomaly_score_threshold'])
        self.data_loader = DataLoader(config)
        self.output_path = Path(config['data']['organised_data_folder_path']) / config['data']['isolation_forest_folder_name']

        self.output_path.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def __intersection(list1, list2):
        # Utility: intersection of two lists
        return list(set(list1) & set(list2))


    def _save_results_to_file(self, model_name, shakespeare_texts_names, anomaly_indices, summa_indices):
        filepath = self.output_path / f"{model_name}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("++++++++++++++++++++++++++++++++++++\n")
            f.write("Isolation Forest Anomalies:\n")
            for idx in anomaly_indices:
                f.write(f"{shakespeare_texts_names[idx]} {idx}\n")
            f.write("\nSummation Ranking Anomalies:\n")
            for idx in summa_indices:
                f.write(f"{shakespeare_texts_names[idx]} {idx}\n")
            f.write("++++++++++++++++++++++++++++++++++++\n")

        self.logger.info(f"Anomaly results saved to {filepath}")

    def analyze(self, model_name):
        """
        Runs Isolation Forest on the DTW distance matrix and analyzes anomalies.

        Args:
            model_name : Model name

        Returns:
            tuple: (summa, scores, predictions, rank)
        """
        dtw_matrix = np.array(self.data_loader.get_dtw(model_name))
        shakespeare_texts_names = self.data_loader.get_shakespeare_included_text_names(model_name)
        lines = self.data_loader.get_text_to_classify()

        if not lines:
            self.logger.warn("Warning: No lines found in text to classify!")

        clf = IsolationForest(n_estimators=self.n_estimators, warm_start=True)
        clf.fit(dtw_matrix)
        y_pred_train = clf.predict(dtw_matrix)
        scores = clf.decision_function(dtw_matrix)

        # Find indices below anomaly score threshold
        anomaly_indices = np.where(scores < self.anomaly_score_threshold)[0]
        anomalies = [shakespeare_texts_names[i] for i in anomaly_indices]
        rank = len(self.__intersection(anomalies, lines))

        # Summation-based ranking normalized
        summa = np.sum(dtw_matrix, axis=1)
        summa = summa / np.sum(summa)

        # Find indices where summa exceeds percentile threshold (Anomalies indexes)
        summa_indices = np.where(summa > np.percentile(summa, self.percentile_threshold))[0]

        # Get the names of all Shakespeare texts that had high total DTW distance (i.e., are outliers), and store them - commented for now
        # those are the anomalies texts
        # summa_anomalies = [shakespeare_texts_names[i] for i in summa_indices]

        # Logging into file
        self._save_results_to_file(model_name, shakespeare_texts_names, anomaly_indices, summa_indices)

        return summa, scores, y_pred_train, rank
