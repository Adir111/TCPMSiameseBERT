import sys
import matplotlib
import wandb

# ♦ Use an interactive backend if we are inside IPython / Colab; otherwise fall
#   back to a non-GUI backend so the code still works on headless nodes.
if "ipykernel" in sys.modules:                       # Jupyter / Colab / VSCode-NB
    matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
else:                                                # headless script / CI
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# UMAP is optional; only import if available so the rest of the library still works if it isn't
try:
    import umap  # noqa: F401
    _HAS_UMAP = True
except ImportError:  # pragma: no cover
    _HAS_UMAP = False

from .logger import WrappedWandbLogger


class DataVisualizer:
    def __init__(self, logger):
        self.logger = logger
        self._is_wandb = isinstance(logger, WrappedWandbLogger)

    def _finalize_plot(self, label, model_name):
        fig = plt.gcf()

        if self._is_wandb:
            key = f"{model_name}/{label}" if model_name else label
            self.logger.log({key: wandb.Image(fig)})

        # interactive backends actually render the figure; Agg just ignores it
        plt.show(block=False)
        plt.close(fig)

    def plot_metric(
        self,
        x=None,
        y_series=None,
        title="",
        x_label="",
        y_label="",
        legend_labels=None,
        *,
        model_name=None
    ):
        plt.figure()

        if y_series is not None:
            for idx, y in enumerate(y_series):
                label = legend_labels[idx] if legend_labels else None
                plt.plot(x if x else range(len(y)), y, label=label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if legend_labels:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(title, model_name)

    def display_loss_plot(self, history, model_name):
        self.plot_metric(
            y_series=[history.history["loss"], history.history["val_loss"]],
            title=f"Loss per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Loss",
            legend_labels=["Training Loss", "Validation Loss"],
            model_name=model_name
        )

    def display_accuracy_plot(self, history, model_name):
        self.plot_metric(
            y_series=[history.history["accuracy"], history.history["val_accuracy"]],
            title=f"Accuracy per Epoch for {model_name}",
            x_label="Epoch",
            y_label="Accuracy",
            legend_labels=["Training Accuracy", "Validation Accuracy"],
            model_name=model_name
        )

    def display_signal_plot(self, signal, text_name, model_name):
        self.plot_metric(
            y_series=[signal],
            title=f"Signal Representation for {text_name} using {model_name}",
            x_label="Batch Index",
            y_label="Mean Prediction Value",
            legend_labels=[model_name],
            model_name=model_name
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
        self._finalize_plot("T-SNE Visualization", model_name="Classifier Encoder Model")

    # ------------------------------------------------------------------ #
    # Embedding projections -------------------------------------------- #
    # ------------------------------------------------------------------ #
    def _scatter_embeddings(self, embedded, labels, title: str):
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
        plt.xlabel("Dim‑1")
        plt.ylabel("Dim‑2")
        plt.colorbar(scatter, label="Label / Cluster")
        plt.grid(True)
        plt.tight_layout()
        self._finalize_plot(title, model_name="Classifier Encoder Model")

    def tsne(self, embeddings, labels, *, perplexity: int = 30, n_iter: int = 1000,
             random_state: int = 42, title: str | None = None):
        """Compute **t‑SNE** projection and display it."""
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    metric="euclidean", random_state=random_state)
        embedded = tsne.fit_transform(embeddings)
        self._scatter_embeddings(embedded, labels, title or "t‑SNE Projection")

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
        >>> viz.plot_embedding(emb, lbl, method="tsne", perplexity=40, title="My t‑SNE")
        >>> viz.plot_embedding(emb, lbl, method="umap", n_neighbors=20, min_dist=0.05)
        """
        method = method.lower()
        if method == "tsne":
            self.tsne(embeddings, labels, title=title, **kwargs)
        elif method == "umap":
            self.umap(embeddings, labels, title=title, **kwargs)
        else:
            raise ValueError(f"Unknown embedding method '{method}'. Use 'tsne' or 'umap'.")
