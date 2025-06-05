from pathlib import Path

import numpy as np
from sklearn_extra.cluster import KMedoids

from PhaseB.bert_siamese_authorship_verification.src.data_loader import DataLoader
from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json


class Clustering:
    _instance = None # Singleton instance

    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(Clustering, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config, logger):
        if self._initialized:
            return  # Avoid reinitialization
        self._initialized = True

        self.logger = logger
        self.data_loader = DataLoader(config)

        self.clustering_algorithm = config['clustering']['algorithm']
        self.n_clusters = config['clustering'][self.clustering_algorithm]['n_clusters']
        self.random_state = config['clustering'][self.clustering_algorithm]['random_state']

        self.output_file_path = Path(config['data']['organised_data_folder_path']) / config['data']['clustering_output_file']

    def cluster_results(self):
        """
        Runs clustering on the anomaly score matrix aggregated across all models.

        Returns:
            tuple: (cluster_labels, medoid_indices)
        """
        all_scores_dict = self.data_loader.get_isolation_forest_results()
        self.logger.info(f"Running clustering algorithm: {self.clustering_algorithm}")

        # --- Step 1: Build matrix ---
        text_names = sorted(
            set.intersection(*(set(scores.keys()) for scores in all_scores_dict.values()))
        )
        model_names = sorted(all_scores_dict.keys())
        score_matrix = np.array([
            [all_scores_dict[model][text] for model in model_names]
            for text in text_names
        ])

        # --- Step 2: Clustering ---
        if self.clustering_algorithm == 'k-medoids':
            clustering_model = KMedoids(
                n_clusters=self.n_clusters,
                metric='euclidean',
                random_state=self.random_state
            )
            clustering_model.fit(score_matrix)
            cluster_labels = clustering_model.labels_
            medoid_indices = clustering_model.medoid_indices_

            # --- Step 3: Save results ---
            self.__save_results_to_file(
                cluster_labels=cluster_labels,
                medoid_indices=medoid_indices,
                text_names=text_names,
                model_names=model_names
            )
            self.logger.info("✅ Finished K-Medoids clustering.")

            return cluster_labels, medoid_indices

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")


    def __save_results_to_file(self, cluster_labels, medoid_indices, text_names, model_names):
        """
        Save the clustering results to a JSON file.

        Args:
            cluster_labels: array-like
            medoid_indices: array-like
            text_names: list of document names
            model_names: list of model names used in the feature space
        """
        results = {
            "algorithm": self.clustering_algorithm,
            "n_clusters": self.n_clusters,
            "model_features_used": model_names,
            "cluster_assignments": {
                text: int(label)
                for text, label in zip(text_names, cluster_labels)
            },
            "medoid_texts": [
                text_names[i] for i in medoid_indices
            ] if medoid_indices is not None else None
        }

        save_to_json(results, self.output_file_path, "Clustering Results")
