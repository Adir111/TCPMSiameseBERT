# import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import triang
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys

from models.keras_bert_siamese import build_keras_siamese_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models.pytorch_bert_siamese import BertSiameseNetwork
from config.get_config import get_config
from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor
from src.dtw import compute_dtw_distance
from src.isolation_forest import AnomalyDetector
from src.clustering import perform_kmedoids_clustering


def preprocess_and_divide_text(text):
    config = get_config()
    chunk_size = config['training']['batch_size'] // config['training']['chunk_factor']
    preprocessor = TextPreprocessor()

    tokens = preprocessor.tokenize_text(text)

    chunks = preprocessor.divide_tokens_into_chunks(tokens, chunk_size)
    return chunks


def preprocess_and_divide_impostor_pair(impostor_1, impostor_2):
    config = get_config()
    chunk_size = config['training']['batch_size'] // config['training']['chunk_factor']
    preprocessor = TextPreprocessor()

    tokens_1 = preprocessor.tokenize_text(impostor_1)
    tokens_2 = preprocessor.tokenize_text(impostor_2)

    chunks_1 = preprocessor.divide_tokens_into_chunks(tokens_1, chunk_size)
    chunks_2 = preprocessor.divide_tokens_into_chunks(tokens_2, chunk_size)

    x1_labels, y1_labels, x2_labels, y2_labels = preprocessor.create_model_x_y(chunks_1, chunks_2)

    enc1 = preprocessor.encode_tokenized_chunks(x1_labels, config['bert']['maximum_sequence_length'])
    enc2 = preprocessor.encode_tokenized_chunks(x2_labels, config['bert']['maximum_sequence_length'])

    x = [
        enc1["input_ids"], enc1["attention_mask"],
        enc2["input_ids"], enc2["attention_mask"]
    ]
    y = np.array(y1_labels + y2_labels)

    return x, y


def train_network_keras(config, x, y, pair_name):
    trained_models_path = config['data']['trained_models_path']
    os.makedirs(trained_models_path, exist_ok=True)
    model_path = os.path.join(trained_models_path, f"model_{pair_name}.h5")

    print("-------------------------")
    print("Started training model:", pair_name)
    model = build_keras_siamese_model()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4,
                                   patience=config['training']['early_stopping_patience'])
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # Todo: Fix validation split
    history = model.fit(x, y, epochs=config['training']['epochs'],
                        validation_split=0.2,
                        callbacks=[early_stopping, checkpoint],
                        verbose=1)
    model.save(model_path)
    print("Accuracy:", history.history['accuracy'][-1])
    print("Loss:", history.history['loss'][-1])
    print("Validation Accuracy:", history.history['val_accuracy'][-1])
    print("Finished training model:", pair_name)
    print("-------------------------")

    return model, history


def display_training_results(history):
    print("\n[INFO] Training History Summary:")
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch + 1}: "
              f"Loss = {history.history['loss'][epoch]:.4f}, "
              f"Accuracy = {history.history['accuracy'][epoch]:.4f}, "
              f"Val_Loss = {history.history['val_loss'][epoch]:.4f}, "
              f"Val_Accuracy = {history.history['val_accuracy'][epoch]:.4f}")


def display_loss_plot(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_accuracy_plot(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_trained_networks():
    config = get_config()
    trained_networks_path = config['data']['trained_models_path']
    os.makedirs(trained_networks_path, exist_ok=True)

    trained_networks = []
    for model_file in os.listdir(trained_networks_path):
        if model_file.endswith(".h5"):
            model_path = os.path.join(trained_networks_path, model_file)
            model = tf.keras.models.load_model(model_path)
            trained_networks.append({
                "model": model,
                "model_name": model_file
            })
    return trained_networks


def display_signal_plot(signal, model_name):
    plt.plot(signal, label=model_name)
    plt.title(f"Signal Representation for {model_name}")
    plt.xlabel("Chunk Index")
    plt.ylabel("Mean Prediction Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def classify_text(text_to_classify, trained_networks, batch_size):
    text_chunks = preprocess_and_divide_text(text_to_classify)
    # Train X networks on X impostor pairs or load the saved trained X networks
    # Classify in each trained network
    all_signal_representations = []
    for network in trained_networks:
        # Todo: Model expects 4 inputs instead of 2 (we need one branch)
        model = network['model']
        model_name = network['model_name']
        print(f"[INFO] Classifying text with model {model_name}...")
        # Convert text chunks to numpy array
        text_chunks = np.array(text_chunks)
        # Predict using the trained model
        predictions = np.asarray(model.predict(text_chunks))[:, 0]
        print(f"[INFO] Predictions for {model_name}: {predictions}")

        # Calculate the mean value of each batch's chunks
        signal = [np.mean(predictions[i:i + batch_size]) for i in range(0, len(predictions), batch_size)]
        all_signal_representations.append(signal)
        display_signal_plot(signal, model_name)
    # Perform DTW on the signals (all_signal_representations = [signal_1, signal_2, ...], where each signal is a list of means)
    dtw_matrix = np.zeros((len(all_signal_representations), len(all_signal_representations)))
    for i in range(len(all_signal_representations)):
        for j in range(i + 1, len(all_signal_representations)):
            dtw_distance = compute_dtw_distance(all_signal_representations[i], all_signal_representations[j])
            print(
                f"[INFO] DTW distance between {trained_networks[i]['model_name']} and {trained_networks[j]['model_name']}: {dtw_distance}")
            dtw_matrix[i][j] = dtw_distance

    # Detect anomalies using Isolation Forest
    anomaly_detector = AnomalyDetector()
    anomaly_vector = anomaly_detector.fit_score(dtw_matrix)
    print("[INFO] Anomaly vector:", anomaly_vector)
    return anomaly_vector


def full_procedure_keras():
    config = get_config()
    batch_size = config['training']['batch_size']
    load_trained = config['model'].get('load_trained_models', False)
    trained_networks = load_trained_networks() if load_trained else []

    data_loader = DataLoader(config['data']['processed_impostors_path'])
    cleaned_impostor_data = data_loader.load_cleaned_text_pair()

    if len(trained_networks) == 0:
        for idx, (impostor_1, impostor_2, pair_name) in enumerate(cleaned_impostor_data):
            print(f"[INFO] Training model {idx + 1}/{len(cleaned_impostor_data)} - for impostor pair {pair_name}")
            x, y = preprocess_and_divide_impostor_pair(impostor_1, impostor_2)
            model, history = train_network_keras(config, x, y, pair_name)
            trained_networks.append({
                "model": model,
                "model_name": pair_name
            })
            display_training_results(history)
            display_loss_plot(history)
            display_accuracy_plot(history)
    else:
        print("[INFO] Loaded trained networks:")
        for network in trained_networks:
            print(f"[INFO] Model name: {network['model_name']}")

    print("[INFO] Loading Shakespeare data for testing...")
    tested_collection_texts = DataLoader(config['data']['processed_tested_path'])
    tested_collection_data = tested_collection_texts.load_cleaned_text()
    print("[INFO] Number of texts in tested collection:", len(tested_collection_data))
    anomaly_scores = []
    for text_idx, text in enumerate(tested_collection_data):
        anomaly_vector = classify_text(text, trained_networks, batch_size)
        anomaly_scores.append(anomaly_vector)
    print("[INFO] Anomaly scores for tested collection:", anomaly_scores)
    print("[INFO] Finished processing all texts.")

    # Perform clustering on the anomaly scores
    kmedoids = perform_kmedoids_clustering(anomaly_scores, num_clusters=2)
    print("[INFO] Clustering results:", kmedoids)
    # Visualize the results with t-SNE
    print("[INFO] Visualizing results with t-SNE...")
    anomaly_array = np.array(anomaly_scores)
    if anomaly_array.shape[1] < 2:
        print("[WARN] Not enough variance for t-SNE — using PCA")
        tsne_results = PCA(n_components=2).fit_transform(anomaly_array)
    else:
        tsne = TSNE(n_components=2, perplexity=min(5, len(anomaly_array) - 1), random_state=42)
        tsne_results = tsne.fit_transform(anomaly_array)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("t-SNE Visualization of Tested Texts by Anomaly Scores")
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmedoids, cmap='viridis', s=60, edgecolors='k')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    full_procedure_keras()
