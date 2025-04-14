import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib - so it works on Colab
import matplotlib.pyplot as plt

from utilities.logger import WrappedWandbLogger


class DataVisualizer:
    def __init__(self, logger):
        self.logger = logger
        self._is_wandb = isinstance(logger, WrappedWandbLogger)

    def _finalize_plot(self, label):
        if self._is_wandb:
            self.logger.log({label: self.logger.wandb.Image(plt.gcf())})
        else:
            plt.show()

    def plot_metric(
        self,
        x=None,
        y_series=None,
        title="",
        x_label="",
        y_label="",
        legend_labels=None,
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
