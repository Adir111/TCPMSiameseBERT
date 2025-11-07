"""
DataVisualizer module for generating and saving various plots related to
machine learning model training, embeddings visualization, and anomaly detection.

This module supports:
- Plotting metrics like loss and accuracy over epochs
- Visualizing signals and embeddings (t-SNE and optionally UMAP)
- Clustering results visualization (DBSCAN-based core vs outside points)
- Integration with Weights & Biases for experiment tracking (optional)

Dependencies:
- numpy
- matplotlib
- scikit-learn
- optionally umap-learn (for UMAP projections)
"""

import numpy as np
import matplotlib
from pathlib import Path
from datetime import datetime


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# UMAP is optional; only import if available so the rest of the library still works if it isn't
try:
    import umap  # noqa: F401

    _HAS_UMAP = True
except ImportError:  # pragma: no cover
    _HAS_UMAP = False


class DataVisualizer:
    _instance = None

    def __new__(cls, is_wandb_logger, logger):
        """
        Create a singleton instance of DataVisualizer.
        Prevents multiple initializations.

        Args:
            is_wandb_logger (bool): Whether to log plots to Weights & Biases.
            logger: Logger instance for logging messages.
        """
        if cls._instance is None:
            cls._instance = super(DataVisualizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self, is_wandb_logger, logger):
        """
        Initialize DataVisualizer instance.
        Does nothing if already initialized.

        Args:
            is_wandb_logger (bool): Whether to log plots to Weights & Biases.
            logger: Logger instance for logging messages.
        """
        if self._initialized:
            return  # Prevent reinitialization

        self.logger = logger
        self._is_wandb = is_wandb_logger

        self._initialized = True


    def _finalize_plot(self, label, save_path=None, add_date=True):
        """
        Save current matplotlib figure to file and optionally log to W&B.

        Args:
            label: Label or title used for the filename and logging.
            save_path: Optional directory path to save the figure. Defaults to 'plots/'.
            add_date: Whether to append the current datetime to the filename. Default is True.
        """
        fig = plt.gcf()
        if save_path is None:
            plots_dir = Path("plots")
        else:
            plots_dir = Path(save_path)
        plots_dir.mkdir(parents=True, exist_ok=True)

        if add_date:
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = plots_dir / f"{label}_{current_date}.png"
        else:
            filename = plots_dir / f"{label}.png"

        fig.savefig(filename)

        if self._is_wandb:
            try:
                image = self.logger.Image(str(filename))
                self.logger.log({label: image})
                self.logger.log(f"✅ Logged {label} to W&B")
            except Exception as e:
                self.logger.error(f"[W&B] Failed to log figure: {e}")

        plt.close(fig)


    def plot_metric(self, y_series=None, title="", x_label="", y_label="",
                    legend_labels=None, save_path=None, filename_override=None):
        """
        Plot one or more series of y-values against their indices.

        Args:
            y_series (list of list or np.ndarray): Series to plot.
            title (str): Plot title.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
            legend_labels (list of str): Labels for the legend.
            save_path (str or Path): Optional directory path to save the figure.
            filename_override (str): Optional filename to override the title-based one.
        """
        width = max(6, int(len(title) * 0.1))
        height = 0.75 * width

        plt.figure(figsize=(width, height))

        if y_series is not None:
            for idx, y in enumerate(y_series):
                label = legend_labels[idx] if legend_labels else None
                x = range(len(y))
                plt.plot(x, y, label=label)
                plt.ylim(-0.1, max(y) + 0.5)
                plt.xticks(x)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if legend_labels:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(label=filename_override or title, save_path=save_path, add_date=False)


    def display_loss_plot(self, history, model_name):
        """
        Display training and validation loss over epochs.

        Args:
            history: Keras History object from model.fit.
            model_name (str): Name of the model for the plot title.
        """
        self.plot_metric(
            y_series=[history.history["loss"], history.history["val_loss"]],
            title=f"Loss per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Loss",
            legend_labels=["Training Loss", "Validation Loss"]
        )


    def display_accuracy_plot(self, history, model_name):
        """
        Display training and validation accuracy over epochs.

        Args:
            history: Keras History object from model.fit.
            model_name (str): Name of the model for the plot title.
        """
        self.plot_metric(
            y_series=[history.history["accuracy"], history.history["val_accuracy"]],
            title=f"Accuracy per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Accuracy",
            legend_labels=["Training Accuracy", "Validation Accuracy"]
        )


    def display_signal_plot(self, signal, text_name, model_name, save_path=None):
        """
        Plot a single signal series and save it with a descriptive title.

        Args:
            signal (list or np.ndarray): Signal values to plot.
            text_name (str): Name of the text source.
            model_name (str): Model name used in the title.
            save_path (str or Path): Optional path to save the figure.
        """
        self.plot_metric(
            y_series=[signal],
            title=f"Signal Representation for {text_name} using {model_name}",
            x_label="Batch Index",
            y_label="Mean Prediction Value",
            legend_labels=[model_name],
            save_path=save_path,
            filename_override=text_name
        )


    def display_tsne_plot(self, tsne_results, cluster_labels):
        """
        Plot a 2D scatter plot of t-SNE results colored by cluster labels.

        Args:
            tsne_results (np.ndarray): 2D coordinates from t-SNE.
            cluster_labels (list or np.ndarray): Cluster labels for coloring.
        """
        plt.figure(figsize=(10, 6))
        plt.title("t-SNE Visualization of Tested Texts by Anomaly Scores")
        plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=cluster_labels,
            cmap='viridis',
            s=60,
            edgecolors='k'
        )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot("T-SNE Visualization")


    # ------------------------------------------------------------------ #
    # Embedding projections -------------------------------------------- #
    # ------------------------------------------------------------------ #
    def _scatter_embeddings(self, embedded, labels, title: str, medoid_indices=None):
        """
        Helper to create a scatter plot for 2D embeddings with optional medoid highlighting.

        Args:
            embedded (np.ndarray): 2D embedded points.
            labels (list or np.ndarray): Color labels for points.
            title (str): Plot title.
            medoid_indices (list or np.ndarray, optional): Indices of medoid points to highlight.
        """
        plt.figure(figsize=(10, 6))
        # plt.title(title)
        scatter = plt.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=labels,
            cmap="viridis",
            s=60,
            edgecolors="k",
        )
        # if medoid_indices is not None:
        #     plt.scatter(
        #         embedded[medoid_indices, 0],
        #         embedded[medoid_indices, 1],
        #         c='red',
        #         s=180,
        #         edgecolors='white',
        #         marker='*',
        #         label='Medoids'
        #     )
        #     plt.legend()

        plt.xlabel("Dim‑1")
        plt.ylabel("Dim‑2")
        # plt.colorbar(scatter, label="Label / Cluster")
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(title)


    def tsne(self, embeddings, labels, *, perplexity: int = 30, n_iter: int = 1000,
             random_state: int = 42, title: str | None = None, medoid_indices=None):
        """
        Compute and plot t-SNE embedding of data points.

        Args:
            embeddings (np.ndarray): High-dimensional data points.
            labels (list or np.ndarray): Labels for coloring points.
            perplexity (int): t-SNE perplexity parameter.
            n_iter (int): Number of t-SNE iterations.
            random_state (int): Random seed.
            title (str, optional): Plot title.
            medoid_indices (list or np.ndarray, optional): Indices to highlight as medoids.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    metric="euclidean", random_state=random_state)
        embedded = tsne.fit_transform(embeddings)
        self._scatter_embeddings(
            embedded,
            labels,
            title or "t‑SNE Projection",
            medoid_indices=medoid_indices
        )


    def umap(self, embeddings, labels, *, n_neighbors: int = 15, min_dist: float = 0.1,
             metric: str = "euclidean", random_state: int = 42, title: str | None = None):
        """
        Compute and plot UMAP embedding of data points (requires umap-learn).

        Args:
            embeddings (np.ndarray): High-dimensional data points.
            labels (list or np.ndarray): Labels for coloring points.
            n_neighbors (int): Number of neighbors for UMAP.
            min_dist (float): Minimum distance parameter for UMAP.
            metric (str): Distance metric for UMAP.
            random_state (int): Random seed.
            title (str, optional): Plot title.

        Raises:
            ImportError: If umap-learn is not installed.
        """
        if not _HAS_UMAP:
            raise ImportError("UMAP visualization requested, but 'umap-learn' is not installed."
                              "Run `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=random_state)
        embedded = reducer.fit_transform(embeddings)
        self._scatter_embeddings(embedded, labels, title or "UMAP Projection")


    # unified wrapper --------------------------------------------------- #
    def plot_embedding(self, embeddings, labels, *, method: str = "tsne", title: str | None = None, **kwargs):
        """
        Unified method to plot embeddings using t-SNE or UMAP.

        Args:
            embeddings (np.ndarray): Data points to embed.
            labels (list or np.ndarray): Labels for coloring points.
            method (str): 'tsne' or 'umap' to select embedding method.
            title (str, optional): Plot title.
            **kwargs: Additional parameters for embedding method.

        Raises:
            ValueError: If unknown method is passed.
        """
        method = method.lower()
        if method == "tsne":
            self.tsne(embeddings, labels, title=title, **kwargs)
        elif method == "umap":
            self.umap(embeddings, labels, title=title, **kwargs)
        else:
            raise ValueError(f"Unknown embedding method '{method}'. Use 'tsne' or 'umap'.")


    def plot_core_vs_outside_from_score_matrix(self, score_matrix, title):
        """
        Perform t-SNE and DBSCAN clustering on score matrix,
        then plot and return indices of core and outside clusters.

        Args:
            score_matrix (np.ndarray): Anomaly scores matrix (samples x features).
            title (str): Title for the plot.

        Returns:
            core_indices (list[int]): Indices belonging to core cluster.
            outside_indices (list[int]): Indices belonging to outside cluster.
        """
        # Step 1: t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(score_matrix)

        # Step 2: Normalize for better clustering
        scaled = StandardScaler().fit_transform(embeddings)

        # Step 3: DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(scaled)

        # Step 4: Largest cluster is CORE
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            self.logger.warn("⚠️ DBSCAN failed to find any CORE cluster.")
            return [], []

        core_label = unique_labels[np.argmax(counts)]
        core_indices = np.where(labels == core_label)[0]
        outside_indices = np.where(labels != core_label)[0]

        # Step 5: Plot
        self.__display_core_vs_outside_plot(embeddings, core_indices, outside_indices, title)
        return core_indices, outside_indices


    def __display_core_vs_outside_plot(self, embeddings_2d, core_indices, outside_indices, title):
        """
        Internal helper to plot core (green) vs outside (orange) points.

        Args:
            embeddings_2d (np.ndarray): 2D embeddings coordinates.
            core_indices (list[int]): Indices of core points.
            outside_indices (list[int]): Indices of outside points.
            title (str): Plot title.
        """
        plt.figure(figsize=(10, 6))
        plt.title(title)

        # Plot outside (suspicious) in orange
        plt.scatter(
            embeddings_2d[outside_indices, 0],
            embeddings_2d[outside_indices, 1],
            c="orange",
            label="Suspicious",
            edgecolors="k"
        )

        # Plot core in green
        plt.scatter(
            embeddings_2d[core_indices, 0],
            embeddings_2d[core_indices, 1],
            c="green",
            label="Shakespeare (CORE)",
            edgecolors="k"
        )

        plt.xlabel("Dim‑1")
        plt.ylabel("Dim‑2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(title)


    def display_dtw_heatmap(self, reordered_matrix, model_name, is_sorted, save_path=None):
        """
        Displays and saves a DTW heatmap plot using the reordered distance matrix.

        Args:
            reordered_matrix: The reordered DTW distance matrix.
            model_name: Name of the model (used for naming the saved file).
            is_sorted: Whether the matrix was sorted within clusters.
            save_path: Optional path to save the plot. Defaults to None.
        """
        title = f"DTW Clustered Heatmap for {model_name} ({'Sorted' if is_sorted else 'Unsorted'})"

        plt.figure(figsize=(8, 7))
        plt.imshow(reordered_matrix, interpolation='nearest', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()

        label = f"dtw_clustered_heatmap_{'sorted' if is_sorted else 'unsorted'}"
        self._finalize_plot(label, save_path=save_path, add_date=False)

    def plot_line_graph(self, x_values, y_values, title, x_label, y_label, output_name):
        """
        Plots and saves a simple line graph.
        """
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, marker="o", linestyle="-")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{output_name}.png")
        plt.savefig(output_path)
        plt.close()
