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
        if cls._instance is None:
            cls._instance = super(DataVisualizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, is_wandb_logger, logger):
        if self._initialized:
            return  # Prevent reinitialization

        self.logger = logger
        self._is_wandb = is_wandb_logger

        self._initialized = True

    def _finalize_plot(self, label):
        fig = plt.gcf()
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = plots_dir / f"{label}_{current_date}.png"
        fig.savefig(filename)

        if self._is_wandb:
            try:
                image = self.logger.Image(str(filename))
                self.logger.log({label: image})
                self.logger.log(f"✅ Logged {label} to W&B")
            except Exception as e:
                self.logger.error(f"[W&B] Failed to log figure: {e}")

        plt.close(fig)

    def plot_metric(
            self,
            y_series=None,
            title="",
            x_label="",
            y_label="",
            legend_labels=None,
    ):

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
        self._finalize_plot(title)

    def display_loss_plot(self, history, model_name):
        self.plot_metric(
            y_series=[history.history["loss"], history.history["val_loss"]],
            title=f"Loss per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Loss",
            legend_labels=["Training Loss", "Validation Loss"]
        )

    def display_accuracy_plot(self, history, model_name):
        self.plot_metric(
            y_series=[history.history["accuracy"], history.history["val_accuracy"]],
            title=f"Accuracy per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Accuracy",
            legend_labels=["Training Accuracy", "Validation Accuracy"]
        )

    def display_signal_plot(self, signal, text_name, model_name):
        self.plot_metric(
            y_series=[signal],
            title=f"Signal Representation for {text_name} using {model_name}",
            x_label="Batch Index",
            y_label="Mean Prediction Value",
            legend_labels=[model_name]
        )

    def display_tsne_plot(self, tsne_results, cluster_labels):
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
        """Shared scatter‑plot helper for both t‑SNE and UMAP."""
        plt.figure(figsize=(10, 6))
        plt.title(title)
        scatter = plt.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=labels,
            cmap="viridis",
            s=60,
            edgecolors="k",
        )
        if medoid_indices is not None:
            plt.scatter(
                embedded[medoid_indices, 0],
                embedded[medoid_indices, 1],
                c='red',
                s=180,
                edgecolors='white',
                marker='*',
                label='Medoids'
            )
            plt.legend()

        plt.xlabel("Dim‑1")
        plt.ylabel("Dim‑2")
        plt.colorbar(scatter, label="Label / Cluster")
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(title)


    def tsne(self, embeddings, labels, *, perplexity: int = 30, n_iter: int = 1000,
             random_state: int = 42, title: str | None = None, medoid_indices=None):
        """Compute **t‑SNE** projection and display it."""
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
        """Compute **UMAP** projection and display it (if `umap-learn` installed)."""
        if not _HAS_UMAP:
            raise ImportError("UMAP visualization requested, but 'umap-learn' is not installed."
                              "Run `pip install umap-learn`.")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=random_state)
        embedded = reducer.fit_transform(embeddings)
        self._scatter_embeddings(embedded, labels, title or "UMAP Projection")


    # unified wrapper --------------------------------------------------- #
    def plot_embedding(self, embeddings, labels, *, method: str = "tsne", title: str | None = None, **kwargs):
        """Unified front‑end: call with `method="tsne"` or `method="umap"`.

        Examples
        --------
        # >>> viz.plot_embedding(emb, lbl, method="tsne", perplexity=40, title="My t‑SNE")
        # >>> viz.plot_embedding(emb, lbl, method="umap", n_neighbors=20, min_dist=0.05)
        """
        method = method.lower()
        if method == "tsne":
            self.tsne(embeddings, labels, title=title, **kwargs)
        elif method == "umap":
            self.umap(embeddings, labels, title=title, **kwargs)
        else:
            raise ValueError(f"Unknown embedding method '{method}'. Use 'tsne' or 'umap'.")

    def plot_core_vs_outside_from_score_matrix(self, score_matrix):
        """
        Runs t-SNE, DBSCAN, and plots CORE vs suspicious points.

        Args:
            score_matrix (ndarray): Anomaly scores matrix (n_samples x n_features).

        Returns:
            core_indices (List[int]), outside_indices (List[int])
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
        self.__display_core_vs_outside_plot(embeddings, core_indices, outside_indices)
        return core_indices, outside_indices


    def __display_core_vs_outside_plot(self, embeddings_2d, core_indices, outside_indices):
        """
        Display a 2D scatter plot showing CORE (green) vs Outside (orange).
        """
        plt.figure(figsize=(10, 6))
        plt.title("Detected CORE vs Outside using DBSCAN on t-SNE Embedding")

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
        self._finalize_plot("CORE vs Outside")