import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys

from models.keras_bert_siamese import build_keras_siamese_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pytorch_bert_siamese import BertSiameseNetwork
from config.get_config import get_config
from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor
from src.dtw import compute_dtw_distance
from src.isolation_forest import AnomalyDetector
from src.clustering import perform_kmedoids_clustering


def preprocess_and_divide(impostor_1, impostor_2):
    config = get_config()
    chunk_size = config['training']['batch_size'] // config['training']['chunk_factor']
    preprocessor = TextPreprocessor()

    tokens_1 = preprocessor.tokenize_text(impostor_1)
    tokens_2 = preprocessor.tokenize_text(impostor_2)

    chunks_1 = preprocessor.divide_tokens_into_chunks(tokens_1, chunk_size)
    chunks_2 = preprocessor.divide_tokens_into_chunks(tokens_2, chunk_size)

    x_labels, y_labels, chunks = preprocessor.create_model_x_y(chunks_1, chunks_2)
    return x_labels, y_labels, chunks


def train_network_keras(config, x, y, pair_name):
    trained_models_path = config['data']['trained_models_path']
    os.makedirs(trained_models_path, exist_ok=True)
    model_path = os.path.join(trained_models_path, f"model_{pair_name}.h5")

    print("-------------------------")
    print("Started training model:", pair_name)
    model = build_keras_siamese_model()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4,
                                   patience=config['training']['early_stopping_patience'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    history = model.fit(x, y, epochs=config['training']['epochs'], verbose=1)
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


def classify_text():
    # preprocess_and_divide(text_to_classify)
    # Train X networks on X impostor pairs or load the saved trained X networks
    # for each impostor pair:
    #   preprocess_and_divide(impostor_pair)
    #   train_network_keras()
    #   display_training_results()
    #   display_loss_plot()
    #   display_accuracy_plot()
    # OR load the respective saved trained network
    #   np.asarray(model.predict(text_to_classify))[:, 0]
    #   Calculate the mean value of each batch's chunks
    #   Aggregate all mean values into a single list (signal representation of the text)
    # After all impostor pairs, calculate the DTW distance between each pair of signals
    # Detect anomalies using Isolation Forest
    # Perform clustering on the anomaly scores
    # Visualize the results with t-SNE
    pass


def full_procedure_pytorch():
    # Step 1: Start
    print("[INFO] Loading configuration and initializing device...")
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    wandb_run = None
    if config['wandb']['enabled']:
        print("[INFO] Logging into wandb")
        wandb.login(key=config['wandb']['api_key'])
        wandb_run = wandb.init(project=config['wandb']['project'], config=config,
                               name=f"run_{wandb.util.generate_id()}")

    # Step 2: Preprocessing
    print("[INFO] Loading and preprocessing training data...")
    data_loader = DataLoader(config['data']['processed_impostors_path'])
    cleaned_data = data_loader.load_cleaned_text_pair()
    preprocessor = TextPreprocessor()

    trained_models = []
    trained_models_path = config['data']['trained_models_path']
    os.makedirs(trained_models_path, exist_ok=True)

    batch_size = config['training']['batch_size']
    chunk_factor = config['training']['chunk_factor']
    chunk_size = batch_size // chunk_factor

    for idx, (impostor_1, impostor_2, pair_name) in enumerate(cleaned_data):
        print(f"[INFO] Training model {idx + 1}/{len(cleaned_data)} - for impostor pair {pair_name}")
        model = BertSiameseNetwork().to(device)
        model_path = os.path.join(trained_models_path, f"model_{pair_name}.pt")

        if config['model'].get('load_trained_models', False) and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            trained_models.append(model.state_dict())
            print(f"[INFO] Loaded saved model {idx + 1} from {model_path}")
            continue

        # optimizer = optim.AdamW(model.parameters(), lr=float(config['bert']['learning_rate']))
        # criterion = nn.BCELoss()
        model.train()

        chunks = preprocessor.divide_into_chunk_pair(impostor_1, impostor_2, chunk_size=chunk_size)

        # best_loss = float('inf')
        # patience = config['training']['early_stopping_patience']
        # patience_counter = 0
        for epoch in range(config['training']['epochs']):
            # epoch_loss = 0.0
            for chunk_1, chunk_2 in chunks:
                input_ids1, attention_mask1 = preprocessor.tokenize_chunk(chunk_1)
                input_ids2, attention_mask2 = preprocessor.tokenize_chunk(chunk_2)

                # Move to device
                input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)
                input_ids2, attention_mask2 = input_ids2.to(device), attention_mask2.to(device)

                # Forward pass (imp1, imp2)
                similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)

                # Convert similarity distance to a similarity score in [0, 1]
                score = torch.exp(-similarity).clamp(min=0.0, max=1.0)

                # Determine label based on chunk origin (impostor 1 = 0, impostor 2 = 1)
                # So each pair is (0, 1) — and the "correct" label is always 1 (they're from different authors)
                # label_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(device)
                #
                # # Compute loss
                # loss = criterion(score, label_tensor)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # epoch_loss = loss.item()

            # avg_loss = epoch_loss / len(chunks)
            # print(f"[EPOCH {epoch+1}] Loss: {avg_loss:.4f}")
            # if avg_loss < best_loss:
            #     best_loss = avg_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print("[INFO] Early stopping triggered.")
            #         break
            if config.get("wandb", {}).get("enabled", False):
                wandb.log({
                    "epoch": epoch + 1,
                    # "loss": avg_loss,
                    "impostor_pair_name": pair_name,
                    "num_chunks": len(chunks)
                })

        trained_models.append(model.state_dict())
        if config['model'].get('save_trained_models', False):
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saved model {idx + 1} to {model_path}")

    print("[INFO] Loading Shakespeare data for testing...")
    tested_collection_texts = DataLoader(config['data']['processed_tested_path'])
    tested_collection_data = tested_collection_texts.load_cleaned_text()

    anomaly_scores = []

    for text_idx, text in enumerate(tested_collection_data):
        print(f"[INFO] Processing text {text_idx + 1}/{len(tested_collection_data)} from tested collection...")
        chunks = preprocessor.divide_into_chunk(text, chunk_size=chunk_size)

        num_batches = len(chunks) // batch_size
        print("[INFO] Number of batches:", num_batches)

        chunks = chunks[:num_batches * batch_size]  # Trim overflow
        print("[INFO] Number of chunks after trimming:", len(chunks))

        rows, cols = num_batches, batch_size
        print("[INFO] Number of rows and columns:", rows, cols)

        signal_representations = []
        amount_of_signals = len(trained_models)  # == Number of trained networks
        dtw_matrix = np.zeros((amount_of_signals, amount_of_signals))

        for model_idx, model_state in enumerate(trained_models):
            print(f"[INFO] Using trained model {model_idx + 1}/{len(trained_models)} for signal extraction...")
            labels_matrix = [[0] * cols for _ in range(rows)]
            model = BertSiameseNetwork().to(device)
            model.load_state_dict(model_state)
            model.eval()

            for i, chunk in enumerate(chunks):
                input_ids, attention_mask = preprocessor.tokenize_chunk(chunk)
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

                # One of the trained networks, it does not matter which - Siamese architecture is based on the idea
                # That the weights of both networks are shared.
                similarity = model.forward_single(input_ids, attention_mask)
                labels_matrix[i // cols][i % cols] = similarity.item()

            signal = [[np.mean(row)] for row in
                      labels_matrix]  # Fastdtw expects each signal to be [[0.1], [0.2], [0.3], ...]

            # Obtain signal representation of each text from each trained network
            signal_representations.append(signal)

        print("[INFO] Computing DTW matrix...")
        for i in range(amount_of_signals):
            for j in range(i + 1, amount_of_signals):
                dtw_distance = compute_dtw_distance(signal_representations[i], signal_representations[j])
                dtw_matrix[i][j] = dtw_matrix[j][i] = dtw_distance

        # Matrix shape is (amount_of_signals, amount_of_signals)
        # Convert to a numpy array because of SKLearn's Isolation Forest.
        dtw_matrix = np.array(dtw_matrix)
        print("[INFO] DTW matrix shape:", dtw_matrix.shape)

        print("[INFO] Running anomaly detection...")
        anomaly_detector = AnomalyDetector()
        anomaly_mask = anomaly_detector.fit_score(dtw_matrix)

        print("[INFO] Anomaly mask shape:", anomaly_mask.shape)
        anomaly_scores.append(anomaly_mask.tolist())  # shape = (amount_of_signals,)

    wandb.log({
        "trained_model_count": len(trained_models),
        "anomaly_scores": anomaly_scores
    })
    print("[INFO] Running clustering on anomaly scores...")
    try:
        clusters = perform_kmedoids_clustering(anomaly_scores, num_clusters=2)
    except ValueError as e:
        print(f"[WARN] Clustering failed: {e}")
        return

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
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', s=60, edgecolors='k')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()

    if config.get("wandb", {}).get("enabled", False):
        wandb.log({"cluster_visualization": wandb.Image(plt)})
        wandb_run.finish()
    plt.show()


if __name__ == "__main__":
    full_procedure_pytorch()
