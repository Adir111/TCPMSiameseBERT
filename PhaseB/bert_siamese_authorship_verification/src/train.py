import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.optimizers import AdamW
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys
from transformers import TFBertModel

from models.keras_bert_siamese import build_keras_siamese_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    encoded_chunks = preprocessor.encode_tokenized_chunks(np.asarray(chunks), config['bert']['maximum_sequence_length'])
    return encoded_chunks


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

    lr = float(config['training']['optimizer']['initial_learning_rate'])
    decay = float(config['training']['optimizer']['learning_rate_decay_factor'])
    clipnorm = float(config['training']['optimizer']['gradient_clipping_threshold'])

    print("-------------------------")
    print("Started training model:", pair_name)
    model = build_keras_siamese_model()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4,
                                   patience=config['training']['early_stopping_patience'])
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')

    optimizer = AdamW(learning_rate=lr, weight_decay=decay, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(x, y, epochs=config['training']['epochs'],
                        validation_split=0.33,
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
            model = tf.keras.models.load_model(model_path, custom_objects={
                'TFBertModel': TFBertModel,
                'AdamW': AdamW
            })
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


def extract_encoder_branch_from_siamese(siamese_model):
    print("[INFO] Siamese Model Layers")
    for i, layer in enumerate(siamese_model.layers):
        print(i, layer.name, layer.output_shape)

    # Access the internal branch model
    # Assumes the siamese_model was created like: out1 = branch(...); out2 = branch(...)
    # And the branch is a Functional or Sequential model named something like 'model'
    branch_model = siamese_model.get_layer(index=4)  # This is the shared branch model
    return Model(inputs=branch_model.input, outputs=branch_model.output)


def classify_text(text_to_classify, trained_networks, batch_size):
    text_chunks = preprocess_and_divide_text(text_to_classify)

    all_signal_representations = []
    for network in trained_networks:
        siamese_model = network['model']
        model_name = network['model_name']
        print(f"[INFO] Classifying text with model {model_name}...")

        # Extract only the encoder branch (shared branch from Siamese model)
        encoder_model = extract_encoder_branch_from_siamese(siamese_model)

        # Predict embeddings or scores
        predictions = np.asarray(encoder_model.predict([text_chunks['input_ids'], text_chunks['attention_mask']]))[:, 0]
        print(f"[INFO] Predictions for {model_name}: {predictions}")

        # Aggregate scores into signal chunks
        signal = [np.mean(predictions[i:i + batch_size]) for i in range(0, len(predictions), batch_size)]
        all_signal_representations.append(signal)

        display_signal_plot(signal, model_name)

    # DTW distance matrix
    dtw_matrix = np.zeros((len(all_signal_representations), len(all_signal_representations)))
    for i in range(len(all_signal_representations)):
        for j in range(i + 1, len(all_signal_representations)):
            dtw_distance = compute_dtw_distance(all_signal_representations[i], all_signal_representations[j])
            print(f"[INFO] DTW distance between {trained_networks[i]['model_name']} and {trained_networks[j]['model_name']}: {dtw_distance}")
            dtw_matrix[i][j] = dtw_distance

    # Anomaly detection
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
    if anomaly_array.shape[0] < 2 or anomaly_array.shape[1] < 2:
        print("[WARN] Not enough data points for PCA/t-SNE visualization.")
        tsne_results = np.zeros((anomaly_array.shape[0], 2))  # dummy 2D points
    else:
        try:
            tsne_results = TSNE(n_components=2).fit_transform(anomaly_array)
        except Exception as e:
            print(f"[WARN] Not enough variance for t-SNE — using PCA: {e}")
            tsne_results = PCA(n_components=2).fit_transform(anomaly_array)

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
