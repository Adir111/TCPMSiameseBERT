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

        self.output_path = Path(config['data']['organised_data_folder_path']) / config['data']['clustering_folder_name']
        self.output_path.mkdir(parents=True, exist_ok=True)

    def cluster_dtw(self, model_name):
        """
        Cluster DTW distance matrix using the specified algorithm.

        Args:
            model_name (str): The model identifier.

        Returns:
            tuple: (cluster_labels, medoid_indices or None)
        """
        dtw_matrix = np.array(self.data_loader.get_dtw(model_name))
        self.logger.info(f"Running clustering algorithm: {self.clustering_algorithm}")

        if self.clustering_algorithm == 'k-medoids':
            clustering_model = KMedoids(
                n_clusters=self.n_clusters,
                metric='precomputed', #  meaning: This matrix already contains pairwise distances; don't compute them again.
                random_state=self.random_state
            )
            clustering_model.fit(dtw_matrix)
            cluster_labels = clustering_model.labels_
            medoid_indices = clustering_model.medoid_indices_

            self.__save_results_to_file(model_name, cluster_labels, medoid_indices)
            self.logger.info("Finished K-Medoids clustering.")
            return cluster_labels, medoid_indices
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")


    def __save_results_to_file(self, model_name, labels, medoid_indices):
        result = {
            "model_name": model_name,
            "algorithm": self.clustering_algorithm,
            "n_clusters": self.n_clusters,
            "cluster_labels": labels.tolist(),
            "medoid_indices": medoid_indices.tolist() if medoid_indices is not None else None
        }

        file_path = self.output_path / f"{model_name}.json"
        save_to_json(result, file_path, f"Clustering results for model '{model_name}'")
