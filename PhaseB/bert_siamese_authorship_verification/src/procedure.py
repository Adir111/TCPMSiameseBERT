import numpy as np
import wandb
from keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import AdamW
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys

from src.model import SiameseBertModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.config_loader import get_config
from utilities.logger import get_logger
from utilities.data_visualizer import DataVisualizer
from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor
from src.dtw import compute_dtw_distance
from src.isolation_forest import AnomalyDetector
from src.clustering import perform_kmedoids_clustering


class Procedure:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(self.config)
        self.data_visualizer = DataVisualizer(self.logger)
        self.preprocessor = TextPreprocessor(self.config)
        self.num_chunks_in_batch = self.config['training']['chunk_factor']
        self.max_token_length = self.config['bert']['maximum_sequence_length']
        self.trained_networks = []

    def load_trained_networks(self):
        trained_networks_path = self.config['data']['trained_models_path']
        os.makedirs(trained_networks_path, exist_ok=True)

        trained_networks = []
        for model_file in os.listdir(trained_networks_path):
            if model_file.endswith(".h5"):
                model_path = os.path.join(trained_networks_path, model_file)

                model_name = model_file.removeprefix("model_").removesuffix("_weights.h5")
                siamese = SiameseBertModel(self.config, model_name)
                model = siamese.build_model()
                model.load_weights(model_path)

                trained_networks.append(siamese)
        return trained_networks

    def preprocess_and_divide_text(self, text, text_name):
        config = get_config()
        batch_size = config['training']['batch_size']
        chunk_to_batch_ratio = config['training']['chunk_factor']
        chunk_size = batch_size // chunk_to_batch_ratio

        tokens = self.preprocessor.tokenize_text(text)
        chunks = self.preprocessor.divide_tokens_into_chunks(tokens, chunk_size)

        self.logger.log({
            f"classification - {text_name} num of tokens": len(tokens),
            f"classification - {text_name} num of chunks": len(chunks),
            f"classification - {text_name} num of batches": len(tokens) // batch_size,
            f"classification - {text_name} chunk size": chunk_size
        })

        encoded_chunks = self.preprocessor.encode_tokenized_chunks(np.asarray(chunks), self.max_token_length)
        return encoded_chunks

    def preprocess_and_divide_impostor_pair(self, impostor_1_texts, impostor_2_texts, pair_name):
        chunk_size = self.config['training']['batch_size'] // self.config['training']['chunk_factor']

        chunks_1 = []
        chunks_2 = []

        for text in impostor_1_texts:
            tokens = self.preprocessor.tokenize_text(text)
            chunks = self.preprocessor.divide_tokens_into_chunks(tokens, chunk_size)
            chunks_1.extend(chunks)

        for text in impostor_2_texts:
            tokens = self.preprocessor.tokenize_text(text)
            chunks = self.preprocessor.divide_tokens_into_chunks(tokens, chunk_size)
            chunks_2.extend(chunks)

        x1_labels, y1_labels, x2_labels, y2_labels = self.preprocessor.create_model_x_y(chunks_1, chunks_2)

        self.logger.log({
            f"Pair {pair_name} - impostor 1 number of chunks": len(chunks_1),
            f"Pair {pair_name} - impostor 2 number of chunks": len(chunks_2),
            f"Pair {pair_name} - x1_labels (chunks after balancing)": len(x1_labels),
            f"Pair {pair_name} - x2_labels (chunks after balancing)": len(x2_labels)
        })

        enc1 = self.preprocessor.encode_tokenized_chunks(x1_labels, self.max_token_length)
        enc2 = self.preprocessor.encode_tokenized_chunks(x2_labels, self.max_token_length)

        x = [
            enc1["input_ids"], enc1["attention_mask"],
            enc2["input_ids"], enc2["attention_mask"]
        ]
        y = np.array(y1_labels + y2_labels)

        return x, y

    def train_network_keras(self, x, y, pair_name):
        save_trained_model = self.config['model']['save_trained_models']

        trained_models_path = self.config['data']['trained_models_path']
        os.makedirs(trained_models_path, exist_ok=True)
        model_path = os.path.join(trained_models_path, f"model_{pair_name}_weights.h5")
        model_checkpoint_path = os.path.join(trained_models_path, f"model_{pair_name}_best_weights.h5")

        lr = float(self.config['training']['optimizer']['initial_learning_rate'])
        decay = float(self.config['training']['optimizer']['learning_rate_decay_factor'])
        clip_norm = float(self.config['training']['optimizer']['gradient_clipping_threshold'])

        self.logger.log(f"-------------------------\nStarted training model: {pair_name}")
        model_object = SiameseBertModel(self.config, pair_name)
        model = model_object.build_model()
        self.logger.log({
            f"Model {pair_name} - Trainable Variables": len(model.trainable_variables),
            f"Model {pair_name} - Total Parameters": model.count_params()
        })
        # Todo: Disable early stopping for now
        # early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4,
        #                                patience=config['training']['early_stopping_patience'])
        checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

        optimizer = AdamW(learning_rate=lr, weight_decay=decay, clipnorm=clip_norm)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.logger.log(model.summary())

        history = model.fit(x, y, epochs=self.config['training']['epochs'],
                            validation_split=0.33,
                            # callbacks=[early_stopping, checkpoint],
                            callbacks=[checkpoint],
                            verbose=1)
        if save_trained_model:
            self.logger.log(f"[INFO] Saving model weights to {model_path}")
            self.logger.save(model_path)
            model.save_weights(model_path)
            wandb.run.summary["trained_model_saved_as"] = model_object.get_model_name()

        for epoch in range(len(history.history['loss'])):
            self.logger.log({
                f"Model {pair_name} - Epoch": epoch + 1,
                f"Model {pair_name} - Training Loss": history.history['loss'][epoch],
                f"Model {pair_name} - Validation Loss": history.history['val_loss'][epoch],
                f"Model {pair_name} - Training Accuracy": history.history['accuracy'][epoch],
                f"Model {pair_name} - Validation Accuracy": history.history['val_accuracy'][epoch]
            })

        self.logger.log(f"Accuracy: {history.history['accuracy'][-1]}")
        self.logger.log(f"Loss: {history.history['loss'][-1]}")
        self.logger.log(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")
        self.logger.log(f"Finished training model: {model_object.get_model_name()}\n-------------------------")

        return model_object, history

    def classify_text(self, text_to_classify, text_name):
        text_chunks = self.preprocess_and_divide_text(text_to_classify, text_name)

        all_signal_representations = []
        for network in self.trained_networks:
            model_name = network.get_model_name()
            self.logger.log(f"[INFO] Classifying text {text_name} with model {model_name}...")

            # Extract only the encoder branch (shared branch from Siamese model)
            encoder_model = network.build_encoder_with_classifier()

            # Predict embeddings or scores
            predictions = np.asarray(encoder_model.predict([text_chunks['input_ids'], text_chunks['attention_mask']]))[:, 0]
            binary_outputs = (predictions > 0.5).astype(int)
            binary_outputs = binary_outputs.flatten().tolist()
            self.logger.log(f"[INFO] Predictions for {model_name}: {predictions}")
            self.logger.log(f"[INFO] Rounded up predictions for {model_name}: {binary_outputs}")

            # Aggregate scores into signal chunks
            signal = [np.mean(binary_outputs[i:i + self.num_chunks_in_batch]) for i in
                      range(0, len(binary_outputs), self.num_chunks_in_batch)]
            all_signal_representations.append(signal)

            self.data_visualizer.display_signal_plot(signal, text_name, model_name)

        # DTW distance matrix
        dtw_matrix = np.zeros((len(all_signal_representations), len(all_signal_representations)))
        for i in range(len(all_signal_representations)):
            for j in range(i + 1, len(all_signal_representations)):
                s1 = np.asarray(all_signal_representations[i]).flatten()
                s2 = np.asarray(all_signal_representations[j]).flatten()
                dtw_distance = compute_dtw_distance(s1, s2)
                self.logger.log(
                    f"[INFO] DTW distance between {self.trained_networks[i].get_model_name()} and {self.trained_networks[j].get_model_name()}: {dtw_distance}")
                dtw_matrix[i][j] = dtw_distance

        # Anomaly detection
        anomaly_detector = AnomalyDetector(self.config['isolation_forest']['number_of_trees'])
        anomaly_vector = anomaly_detector.fit_score(dtw_matrix)
        self.logger.log(f"[INFO] Anomaly vector: {anomaly_vector} for text {text_name}")

        return anomaly_vector

    def full_procedure(self):
        load_trained = self.config['model'].get('load_trained_models', False)
        self.trained_networks = self.load_trained_networks() if load_trained else []

        data_loader = DataLoader(self.config['data']['processed_impostors_path'], self.preprocessor)
        cleaned_impostor_pairs = data_loader.load_impostors()

        if len(self.trained_networks) == 0 or len(cleaned_impostor_pairs) != len(self.trained_networks):
            for idx, (impostor_1_texts, impostor_2_texts, pair_name) in enumerate(cleaned_impostor_pairs):
                self.logger.log(f"[INFO] Training model {idx + 1}/{len(cleaned_impostor_pairs)} - for impostor pair {pair_name}")

                x, y = self.preprocess_and_divide_impostor_pair(impostor_1_texts, impostor_2_texts, pair_name)
                model, history = self.train_network_keras(x, y, pair_name)

                self.trained_networks.append(model)

                self.data_visualizer.display_loss_plot(history, model.get_model_name())
                self.data_visualizer.display_accuracy_plot(history, model.get_model_name())
        else:
            self.logger.log("[INFO] Loaded trained networks:")
            for network in self.trained_networks:
                model_name = network.get_model_name()
                self.logger.log(f"[INFO] Model name: {model_name}")
                self.logger.log_summary("loaded_model", model_name)

        self.logger.log("[INFO] Loading Shakespeare data for testing...")
        tested_collection_texts = DataLoader(self.config['data']['processed_tested_path'], self.preprocessor)
        tested_collection_data = tested_collection_texts.load_tested_collection_text()

        self.logger.log({"Number of texts in tested collection": len(tested_collection_data)})

        anomaly_scores = []
        for text_idx, entry in enumerate(tested_collection_data):
            text_name, text = entry
            anomaly_vector = self.classify_text(text, text_name)
            anomaly_scores.append(anomaly_vector)
            self.logger.log({
                f"Anomaly vector of text - {text_name}": self.logger.Histogram(anomaly_vector)
            })

        self.logger.log("[INFO] Finished processing all texts.")
        self.logger.log({"All Anomaly Scores": anomaly_scores})

        # Perform clustering on the anomaly scores
        kmedoids = perform_kmedoids_clustering(anomaly_scores, num_clusters=2)
        if kmedoids is None:
            self.logger.log("[ERROR] Clustering failed. Not enough data points.")
            return

        self.logger.log({"Clustering Labels": kmedoids})

        # Visualize the results with t-SNE
        self.logger.log("[INFO] Visualizing results with t-SNE...")
        anomaly_array = np.array(anomaly_scores)
        if anomaly_array.shape[0] < 2 or anomaly_array.shape[1] < 2:
            self.logger.log("[WARN] Not enough data points for PCA/t-SNE visualization.")
            tsne_results = np.zeros((anomaly_array.shape[0], 2))  # dummy 2D points
        else:
            try:
                tsne_results = TSNE(n_components=2).fit_transform(anomaly_array)
            except Exception as e:
                self.logger.log(f"[WARN] Not enough variance for t-SNE — using PCA: {e}")
                tsne_results = PCA(n_components=2).fit_transform(anomaly_array)

        self.data_visualizer.display_tsne_plot(tsne_results, kmedoids)

        self.logger.finish()


if __name__ == "__main__":
    Procedure().full_procedure()
