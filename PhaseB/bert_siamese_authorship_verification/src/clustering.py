from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn_extra.cluster import KMedoids

from .data_loader import DataLoader
from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json, DataVisualizer


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
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)

        self.clustering_algorithm = config['clustering']['algorithm']
        self.n_clusters = config['clustering'][self.clustering_algorithm]['n_clusters']
        self.random_state = config['clustering'][self.clustering_algorithm]['random_state']
        self.score_matrix = None
        self.text_names = None
        self.cluster_labels = None
        self.medoid_indices = None

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
        self.text_names = sorted(
            set.intersection(*(set(scores.keys()) for scores in all_scores_dict.values()))
        )
        model_names = sorted(all_scores_dict.keys())
        self.score_matrix = np.array([
            [all_scores_dict[model][text] for model in model_names]
            for text in self.text_names
        ])

        # --- Step 2: Clustering ---
        if self.clustering_algorithm == 'k-medoids':
            clustering_model = KMedoids(
                n_clusters=self.n_clusters,
                metric='euclidean',
                random_state=self.random_state
            )
            clustering_model.fit(self.score_matrix)
            self.cluster_labels = clustering_model.labels_
            self.medoid_indices = clustering_model.medoid_indices_

            # --- Step 3: Save results ---
            self.__save_results_to_file(
                model_names=model_names
            )
            self.logger.info("✅ Finished K-Medoids clustering.")

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")


    def plot_clustering_results(self):
        self.logger.info("📊 Plotting cluster...")
        try:
            self.data_visualizer.plot_embedding(
                self.score_matrix,
                self.cluster_labels,
                method="tsne",
                title="t-SNE of Clusters on Anomaly Score Space",
                medoid_indices=self.medoid_indices
            )
        except Exception as e:
            self.logger.warn(f"⚠️ Failed to visualize t-SNE: {e}")


    def print_cluster_assignments(self):
        """
        Print which texts are in each cluster, marking medoids for inspection/debugging.
        """
        medoid_indices = set(self.medoid_indices.tolist()) if self.medoid_indices is not None else set()
        clusters = defaultdict(list)

        for idx, (text, label) in enumerate(zip(self.text_names, self.cluster_labels)):
            is_medoid = idx in medoid_indices
            clusters[label].append((text, is_medoid))

        self.logger.info("📦 Cluster Assignments (⭐ Stars are medoids):")
        for label in sorted(clusters):
            self.logger.info(f"\n🟩 Cluster {label} ({len(clusters[label])} texts):")
            for text, is_medoid in clusters[label]:
                marker = " ⭐" if is_medoid else ""
                self.logger.info(f"   - {text}{marker}")


    def __save_results_to_file(self, model_names):
        """
        Save the clustering results to a JSON file.

        Args:
            model_names: list of model names used in the feature space
        """
        results = {
            "algorithm": self.clustering_algorithm,
            "n_clusters": self.n_clusters,
            "model_features_used": model_names,
            "cluster_assignments": {
                text: int(label)
                for text, label in zip(self.text_names, self.cluster_labels)
            },
            "medoid_texts": [
                self.text_names[i] for i in self.medoid_indices
            ] if self.medoid_indices is not None else None
        }

        save_to_json(results, self.output_file_path, "Clustering Results")
