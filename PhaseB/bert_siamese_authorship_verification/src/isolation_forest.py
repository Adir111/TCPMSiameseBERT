import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

from .data_loader import DataLoader
from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json


class DTWIsolationForest:
    _instance = None  # Singleton instance

    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(DTWIsolationForest, cls).__new__(cls)
        return cls._instance

    def __init__(self, config, logger):
        # Prevent reinitialization in singleton pattern
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.logger = logger
        self.n_estimators = int(config["isolation_forest"]['number_of_trees'])
        self.percentile_threshold = float(config["isolation_forest"]['percentile_threshold'])
        self.anomaly_score_threshold = float(config["isolation_forest"]['anomaly_score_threshold'])
        self.data_loader = DataLoader(config)
        self.output_path = Path(config['data']['organised_data_folder_path']) / config['data']['isolation_forest']['isolation_forest_folder_name']
        self.all_models_scores_path = Path(config['data']['organised_data_folder_path']) / config['data']['isolation_forest']['all_models_scores_file_name']
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.all_models_scores = {}

        self._initialized = True  # Mark as initialized


    @staticmethod
    def __intersection(list1, list2):
        # Utility: intersection of two lists
        return list(set(list1) & set(list2))

    import json

    def __save_results_to_file(self, model_name, scores, shakespeare_texts_names, anomaly_indices, summa_indices):
        # Create subdirectory for model
        model_output_dir = self.output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # === Text File (Anomaly Report) ===
        text_report_path = model_output_dir / "anomaly_report.txt"
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("++++++++++++++++++++++++++++++++++++\n")
            f.write(f"Model: {model_name}\nNumber of documents: {len(scores)}\n")
            f.write(f"Anomaly score threshold: {self.anomaly_score_threshold}\nPercentile threshold: {self.percentile_threshold}\n")
            f.write(f"Isolation Forest Anomaly Scores (score < threshold [{self.anomaly_score_threshold}]):\n")
            for idx in anomaly_indices:
                name = shakespeare_texts_names[idx]
                score = scores[idx]
                f.write(f"{idx} - {name} - Score: {score:.6f}\n")

            f.write("\nSummation Ranking Anomalies (DTW percentile > ")
            f.write(f"{self.percentile_threshold}):\n")
            for idx in summa_indices:
                f.write(f"{idx} - {shakespeare_texts_names[idx]}\n")

            f.write("++++++++++++++++++++++++++++++++++++\n")

        self.logger.info(f"Text anomaly report saved to {text_report_path}")

        # === JSON File (Full Scores Dictionary) ===
        scores_dict = {name: float(score) for name, score in zip(shakespeare_texts_names, scores)}
        json_path = model_output_dir / "anomaly_scores.json"

        save_to_json(scores_dict, json_path, "Anomaly scores")

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
        self.__save_results_to_file(model_name, scores, shakespeare_texts_names, anomaly_indices, summa_indices)

        self.all_models_scores[model_name] = {
            name: float(score) for name, score in zip(shakespeare_texts_names, scores)
        }

        return summa, scores, y_pred_train, rank


    def save_all_models_scores(self):
        """
        Saves the complete all_models_scores dictionary to a JSON file using the provided save_to_json method.
        """
        save_to_json(self.all_models_scores, self.all_models_scores_path, "All models anomaly scores")
