import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bert_siamese import BertSiameseNetwork
from config.get_config import get_config
from src.data_loader import DataLoader
from src.preprocess import TextPreprocessor


def test_signal_extraction_visual():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a random tested collection text
    data_loader = DataLoader(config['data']['processed_tested_path'])
    tested_collection = data_loader.load_cleaned_text()
    random_text = random.choice(tested_collection)

    # Preprocess
    preprocessor = TextPreprocessor()
    batch_size = config['training']['batch_size']
    chunk_size = config['training']['chunk_size']
    chunks = preprocessor.divide_into_chunk(random_text, chunk_size=chunk_size)
    num_batches = len(chunks) // batch_size
    chunks = chunks[:num_batches * batch_size]
    rows, cols = num_batches, batch_size

    # Load trained models
    trained_models_path = config['data']['trained_models_path']
    current_directory = os.path.join(os.path.dirname(__file__), '..')
    model_dir = os.path.join(current_directory, trained_models_path)
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])

    signal_representations = []

    for model_idx, model_file in enumerate(model_files):
        print(f"[TEST] Using trained model {model_file} for signal extraction...")
        labels_matrix = [[0] * cols for _ in range(rows)]
        model = BertSiameseNetwork().to(device)
        model_path = os.path.join(model_dir, model_file)
        model.load_state_dict(torch.load(model_path))

        model.eval()

        for i, chunk in enumerate(chunks):
            input_ids, attention_mask = preprocessor.tokenize_chunk(chunk)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            similarity = model.forward_single(input_ids, attention_mask)
            labels_matrix[i // cols][i % cols] = similarity.item()

        signal = [np.mean(row) for row in labels_matrix]
        signal_representations.append(signal)

    # Plot each signal
    plt.figure(figsize=(12, 6))
    for i, signal in enumerate(signal_representations):
        plt.plot(signal, label=f"Model {i + 1}")

    plt.title("Signal Representations from Trained Networks")
    plt.xlabel("Batch Index")
    plt.ylabel("Mean Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()