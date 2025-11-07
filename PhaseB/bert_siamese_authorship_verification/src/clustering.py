"""
Performs clustering (K-Medoids, K-Means and Kernel K-Means) on anomaly score data from multiple models.
Handles state management, visualization, and saving clustering results.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from KKMeans import KKMeans

from .data_loader import DataLoader
from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json, DataVisualizer


class Clustering:
    """
    Singleton class for clustering text data using anomaly scores from multiple models.
    """

    _instance = None # Singleton instance

    def __new__(cls, config, logger):
        """
        Implements the singleton pattern to ensure only one instance of Clustering exists.
        Initializes the instance if it doesn't exist yet, otherwise returns the existing one.
        """
        if cls._instance is None:
            cls._instance = super(Clustering, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config, logger):
        """
        Initializes the Clustering singleton with config and logger.

        Args:
            config (dict): Configuration dictionary with the needed clustering configurations.
            logger (Logger): Logger instance for logging info and warnings.
        """
        if self._initialized:
            return  # Avoid reinitialization
        self._initialized = True

        self.logger = logger
        self.data_loader = DataLoader(config)
        self.data_visualizer = DataVisualizer(config['wandb']['enabled'], logger)

        self.clustering_algorithm = config['clustering']['algorithm']
        self.n_clusters = config['clustering']['n_clusters']
        self.random_state = config['clustering']['random_state']
        self.score_matrix = None
        self.text_names = None
        self.cluster_labels = None
        self.core_names = []
        self.outside_names = []

        self.medoid_indices = None # K-Medoids Clustering - Results Related
        self.gamma = config['clustering']['gamma']  # K-Means Variable

        self.output_folder_path = (
                Path(config['data']['organised_data_folder_path']) /
                config['data']['clustering_folder_name'] /
                self.clustering_algorithm.lower()
        )
        self.output_folder_path.mkdir(parents=True, exist_ok=True)


    def update_state_from_result(self, result):
        """
        Update internal state with clustering result.

        Args:
            result (dict): Contains 'score_matrix', 'cluster_labels', 'medoid_indices',
                           and optionally 'text_names'.
        """
        self.score_matrix = result.get("score_matrix")
        self.cluster_labels = result.get("cluster_labels")
        self.medoid_indices = result.get("medoid_indices")
        # text_names usually stays the same, but if provided, update it
        if "text_names" in result:
            self.text_names = result["text_names"]


    def cluster_results(self, increment=None):
        """
        Run clustering on the loaded model scores.

        Args:
            increment (int, optional): If set, performs incremental clustering using top-N models.

        Returns:
            list: List of clustering result dictionaries.
        """
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
                clustering_result = self.__run_clustering(current_models, suffix="_all")
                results.append(clustering_result)

        return results


    def __run_clustering(self, model_names, suffix=""):
        """
        Internal method to perform K-Medoids or K-Means clustering.

        Args:
            model_names (list): Model names used as features.
            suffix (str): Filename suffix for saving results.

        Returns:
            dict: Clustering result data.
        """
        if self.clustering_algorithm == 'k-medoids':
            clustering_model = KMedoids(
                n_clusters=self.n_clusters,
                metric='euclidean',
                random_state=self.random_state
            )
            clustering_model.fit(self.score_matrix)
            self.cluster_labels = clustering_model.labels_
            self.medoid_indices = clustering_model.medoid_indices_

        elif self.clustering_algorithm == 'k-means':
            # Optional Gaussian exponential transformation
            gamma = self.gamma
            if gamma is None:
                raise ValueError("Missing 'gamma' for K-Means clustering.")
            transformed_matrix = np.exp(-gamma * np.square(self.score_matrix))

            clustering_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init='auto'
            )
            clustering_model.fit(transformed_matrix)
            self.cluster_labels = clustering_model.labels_
        elif self.clustering_algorithm == 'kk-means':
            # Kernel K-Means clustering with RBF Kernel
            clustering_model = KKMeans(
                n_clusters=self.n_clusters,
                kernel='rbf'
            )
            clustering_model.fit(self.score_matrix)
            self.cluster_labels = clustering_model.labels_

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}")

        self.__save_results_to_file(model_names, suffix)
        self.logger.info(f"✅ {self.clustering_algorithm.upper()} clustering complete for models: {len(model_names)}")

        return {
            "model_names": model_names,
            "suffix": suffix,
            "score_matrix": self.score_matrix.copy(),
            "cluster_labels": self.cluster_labels.copy(),
            "medoid_indices": self.medoid_indices.copy() if self.medoid_indices is not None else None
        }


    def __save_results_to_file(self, model_names, suffix=""):
        """
        Save clustering results to a JSON file.

        Args:
            model_names (list): Model names used in clustering.
            suffix (str): Suffix for the result file.
        """
        results = {
            "algorithm": self.clustering_algorithm,
            "n_clusters": self.n_clusters,
            "model_features_used": model_names,
            "cluster_assignments": {
                text: int(label)
                for text, label in zip(self.text_names, self.cluster_labels)
            }
        }

        # Include medoids only if available (when ran k-medoids
        if self.medoid_indices is not None:
            results["medoid_texts"] = [
                self.text_names[i] for i in self.medoid_indices
            ]

        safe_suffix = suffix.lstrip("_") or "all"
        file_name = f"clustering_results_{safe_suffix}.json"
        file_path = self.output_folder_path / file_name

        save_to_json(results, file_path, f"Clustering Results {suffix or ''}")


    def __save_core_vs_outside_to_file(self, suffix):
        """
        Save CORE and suspicious text names to a JSON file.

        Args:
            suffix (str): Suffix for the result file.
        """
        results = {
            "core_texts": self.core_names,
            "suspicious_texts": self.outside_names,
        }

        safe_suffix = suffix or "all"
        file_name = f"core_vs_outside_{safe_suffix}.json"
        file_path = self.output_folder_path / file_name

        save_to_json(results, file_path, f"CORE vs Outside Results {suffix or ''}")

        self.logger.info(f"🟢 Saved CORE vs Outside results to {file_path}")


    def plot_clustering_results(self, suffix=""):
        """
        Visualize clustering results using t-SNE.

        Args:
            suffix (str): Optional label for plot title.
        """
        self.logger.info("📊 Plotting cluster...")
        try:
            title = f"Clusters on Anomaly Score Space ({self.clustering_algorithm.upper()})"
            if suffix:
                title += f" ({suffix})"

            self.data_visualizer.plot_embedding(
                self.score_matrix,
                self.cluster_labels,
                method="tsne",
                title=title,
                medoid_indices=self.medoid_indices if self.medoid_indices is not None else None
            )
        except Exception as e:
            self.logger.warn(f"⚠️ Failed to visualize t-SNE: {e}")


    def print_full_clustering_summary(self):
        """
        Print clustering results and highlight medoids, core, and suspicious texts.
        """
        algo_name = self.clustering_algorithm.upper()
        self.logger.log(f"📌 ==================== {algo_name} Clustering Summary ====================")

        if self.cluster_labels is None or self.text_names is None:
            self.logger.warn("⚠️ No clustering results found. Please run clustering first.")
            return

        medoid_indices = set(self.medoid_indices.tolist()) if self.medoid_indices is not None else set()
        clusters = defaultdict(list)

        for idx, (text, label) in enumerate(zip(self.text_names, self.cluster_labels)):
            is_medoid = idx in medoid_indices
            clusters[label].append((text, is_medoid))

        self.logger.info("📦 Cluster Assignments{}:".format(
            " (⭐ Stars are medoids)" if self.clustering_algorithm == 'k-medoids' else ""))
        for label in sorted(clusters):
            self.logger.info(f"🟩 Cluster {label} ({len(clusters[label])} texts):")
            for text, is_medoid in clusters[label]:
                marker = " ⭐" if is_medoid else ""
                self.logger.log(f"   - {text}{marker}")

        self.logger.log("📌 ==================== Core vs Suspicious Summary ====================")

        if not self.core_names and not self.outside_names:
            self.logger.warn("⚠️ No CORE vs Outside data found. Did you run `plot_core_vs_outside()`?")
            return

        self.logger.info(f"✅ CORE Texts (Total: {len(self.core_names)}):")
        for text in self.core_names:
            self.logger.log(f"   - {text}")

        self.logger.info(f"⚠️ Suspicious Texts (Total: {len(self.outside_names)}):")
        for text in self.outside_names:
            self.logger.log(f"   - {text}")


    def plot_core_vs_outside(self, suffix):
        """
        Visualize and store CORE vs suspicious texts using DBSCAN on t-SNE embedding.

        Args:
            suffix (str): Suffix for file and plot title.
        """
        title = "Detected CORE vs Outside using DBSCAN on t-SNE Embedding"
        if suffix:
            title += f" ({suffix})"
        core_indices, outside_indices = self.data_visualizer.plot_core_vs_outside_from_score_matrix(self.score_matrix, title)
        self.core_names = [self.text_names[i] for i in core_indices]
        self.outside_names = [self.text_names[i] for i in outside_indices]

        self.__save_core_vs_outside_to_file(suffix)


    def _plot_cluster_vs_models(self, model_counts, cluster_sizes, cluster_num):
        """
        Uses the DataVisualizer to plot cluster 0 size vs number of models.
        """
        x_label = "Number of Models Used"
        y_label = "Number of fake texts"
        output_name = f"cluster{cluster_num}_size_vs_models"
        logger.info(f"model_counts: {model_counts},\ncluster_sizes: {cluster_sizes},\ncluster_num: {cluster_num}")

        try:
            self.data_visualizer.plot_line_graph(
                x_values=model_counts,
                y_values=cluster_sizes,
                x_label=x_label,
                y_label=y_label,
                output_name=output_name
            )
            self.logger.info(f"📊 Successfully generated Cluster {cluster_num} vs Models plot.")
        except Exception as e:
            self.logger.warn(f"⚠️ Failed to generate Cluster {cluster_num} vs Models plot: {e}")


    def analyze_cluster_labels(self, all_labels, model_counts, cluster_num):
        """
        Analyzes collected cluster labels and plots the size of cluster 0
        as a function of the number of models used.
        """
        if not all_labels or not model_counts:
            self.logger.warn("⚠️ No cluster labels or model counts available for analysis.")
            return

        self.logger.info("🧩 Analyzing collected cluster labels across steps...")
        self.logger.info(f"all labels: {all_labels}, model counts: {model_counts}")

        cluster_sizes_all = []

        for step_idx, labels in enumerate(all_labels):
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            cluster_size = cluster_sizes.get(cluster_num, 0)
            cluster_sizes_all.append(cluster_size)
            self.logger.info(f"Step {step_idx + 1}: Cluster {cluster_num} size = {cluster_size}")

        # Use the dedicated plotting method
        self._plot_cluster_vs_models(model_counts, cluster_sizes_all, cluster_num)


    def get_results(self, increment=None):
        """
        Load previously saved clustering results from JSON files.

        Args:
            increment (int, optional): If incremental clustering was used, load all incremental results.
                                       If None, load the "all_models" result only.

        Returns:
            list: List of clustering result dicts matching `cluster_results` structure.
        """
        results = []

        # Determine which files to load
        json_files = sorted(self.output_folder_path.glob("clustering_results_*.json"))

        # If increment is used, include all incremental results
        for file_path in json_files:
            try:
                data = json.load(open(file_path, "r"))

                # Recover the fields expected in cluster_results
                model_names = data.get("model_features_used")
                cluster_labels = np.array([v for v in data.get("cluster_assignments", {}).values()])
                medoid_texts = data.get("medoid_texts", None)
                medoid_indices = None
                if medoid_texts is not None and self.text_names is not None:
                    medoid_indices = [self.text_names.index(t) for t in medoid_texts]

                # Extract suffix from filename
                suffix = file_path.stem.replace("clustering_results_", "")

                # Build score_matrix if needed
                score_matrix = None
                all_scores_dict = self.data_loader.get_isolation_forest_results()
                if model_names:
                    self.text_names = sorted(
                        set.intersection(*(set(scores.keys()) for scores in all_scores_dict.values()))
                    )
                    score_matrix = np.array([
                        [all_scores_dict[model][text] for model in model_names]
                        for text in self.text_names
                    ])

                results.append({
                    "model_names": model_names,
                    "suffix": suffix,
                    "score_matrix": score_matrix,
                    "cluster_labels": cluster_labels,
                    "medoid_indices": medoid_indices
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load clustering results from {file_path}: {e}")

        # If increment is None, filter for the final "all" result only
        if increment is None:
            results = [r for r in results if r["suffix"].lstrip("_") in ("all", "all_models")]

        return results
