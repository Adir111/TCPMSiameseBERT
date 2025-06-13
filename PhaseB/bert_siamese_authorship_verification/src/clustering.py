import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn_extra.cluster import KMedoids

from .data_loader import DataLoader
from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json, DataVisualizer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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

        self.output_folder_path = Path(config['data']['organised_data_folder_path']) / config['data']['clustering_folder_name']
        self.output_folder_path.mkdir(parents=True, exist_ok=True)


    def update_state_from_result(self, result):
        """
        Update internal clustering state from a clustering result dict.

        Args:
            result (dict): A dictionary containing keys:
                - 'score_matrix'
                - 'cluster_labels'
                - 'medoid_indices'
                - (optional) 'text_names' - if you want to override text_names too
        """
        self.score_matrix = result.get("score_matrix")
        self.cluster_labels = result.get("cluster_labels")
        self.medoid_indices = result.get("medoid_indices")
        # text_names usually stays the same, but if provided, update it
        if "text_names" in result:
            self.text_names = result["text_names"]


    def cluster_results(self, increment=None):
        all_scores_dict = self.data_loader.get_isolation_forest_results()
        model_names = sorted(all_scores_dict.keys())

        self.logger.info(f"Running clustering algorithm: {self.clustering_algorithm}")
        self.logger.info(f"Total models: {len(model_names)}")
        self.logger.info(f"Increment: {increment if increment is not None else 'Not Used'}")

        # Get common text entries across all models
        self.text_names = sorted(
            set.intersection(*(set(scores.keys()) for scores in all_scores_dict.values()))
        )

        def build_matrix(subset_model_names):
            return np.array([
                [all_scores_dict[model][text] for model in subset_model_names]
                for text in self.text_names
            ])

        results = []

        if increment is None:
            # Default: use all models
            self.score_matrix = build_matrix(model_names)
            clustering_result = self.__run_clustering(model_names)
            results.append(clustering_result)
        else:
            # Incremental: use growing subsets
            for i in range(increment, len(model_names) + 1, increment):
                current_models = model_names[:i]
                self.logger.info(f"\n🔁 Clustering with first {i} models")
                self.score_matrix = build_matrix(current_models)
                clustering_result = self.__run_clustering(current_models, suffix=f"_top{i}")
                results.append(clustering_result)

            if len(model_names) % increment != 0:
                # Run final cluster if leftover models remain
                current_models = model_names
                self.logger.info(f"\n🔁 Final clustering with all models")
                self.score_matrix = build_matrix(current_models)
                clustering_result = self.__run_clustering(current_models, suffix=f"_all")
                results.append(clustering_result)

        return results


    def __run_clustering(self, model_names, suffix=""):
        if self.clustering_algorithm == 'k-medoids':
            clustering_model = KMedoids(
                n_clusters=self.n_clusters,
                metric='euclidean',
                random_state=self.random_state
            )
            clustering_model.fit(self.score_matrix)
            self.cluster_labels = clustering_model.labels_
            self.medoid_indices = clustering_model.medoid_indices_

            self.__save_results_to_file(model_names, suffix)
            self.logger.info(f"✅ K-Medoids clustering complete for models: {len(model_names)}")

            return {
                "model_names": model_names,
                "suffix": suffix,
                "score_matrix": self.score_matrix.copy(),
                "cluster_labels": self.cluster_labels.copy(),
                "medoid_indices": self.medoid_indices.copy()
            }
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")


    def __save_results_to_file(self, model_names, suffix=""):
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

        safe_suffix = suffix.lstrip("_") or "all"
        file_name = f"clustering_results_{safe_suffix}.json"

        file_path = self.output_folder_path / file_name

        save_to_json(results, file_path, f"Clustering Results {suffix or ''}")

    def save_core_vs_outside_to_file(self, core_names, outside_names, suffix):
        """
        Saves the CORE and Suspicious text names to a JSON file.
        """
        results = {
            "core_texts": core_names,
            "suspicious_texts": outside_names,
        }

        safe_suffix = suffix or "all"
        file_name = f"core_vs_outside_{safe_suffix}.json"
        file_path = self.output_folder_path / file_name

        save_to_json(results, file_path, f"CORE vs Outside Results {suffix or ''}")

        self.logger.info(f"🟢 Saved CORE vs Outside results to {file_path}")


    def plot_clustering_results(self, suffix=""):
        self.logger.info("📊 Plotting cluster...")
        try:
            title = "t-SNE of Clusters on Anomaly Score Space"
            if suffix:
                title += f" ({suffix})"

            self.data_visualizer.plot_embedding(
                self.score_matrix,
                self.cluster_labels,
                method="tsne",
                title=title,
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


    def plot_core_vs_outside(self):
        # Step 1: t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(self.score_matrix)

        # Step 2: Normalize for better clustering
        scaled = StandardScaler().fit_transform(embeddings)

        # Step 3: DBSCAN to find dense core
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(scaled)

        # Step 4: Assume the largest cluster is CORE
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        core_label = unique_labels[np.argmax(counts)]

        core_indices = np.where(labels == core_label)[0]
        outside_indices = np.where(labels != core_label)[0]

        # Step 5: Plot
        self.data_visualizer.display_core_vs_outside_plot(embeddings, core_indices, outside_indices)

        return [self.text_names[i] for i in core_indices], [self.text_names[i] for i in outside_indices]
