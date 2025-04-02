def full_procedure():
    # Step 1: Start
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.bert_siamese import BertSiameseNetwork
    from config.get_config import get_config
    from src.data_loader import DataLoader
    from src.preprocess import TextPreprocessor
    from src.dtw import compute_dtw_distance
    from src.isolation_forest import AnomalyDetector
    from src.clustering import perform_kmedoids_clustering

    # Step 2: Preprocessing
    print("[INFO] Loading configuration and initializing device...")
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    print("[INFO] CUDA available:", torch.cuda.is_available())
    print("[INFO] CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    print("[INFO] Loading and preprocessing training data...")
    data_loader = DataLoader(config['data']['processed_impostors_path'])
    cleaned_data = data_loader.load_cleaned_text_pair()
    preprocessor = TextPreprocessor()

    trained_models = []
    for idx, (impostor_1, impostor_2) in enumerate(cleaned_data):
        print(f"[INFO] Training model {idx+1}/{len(cleaned_data)} for impostor pair...")
        model = BertSiameseNetwork().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=float(config['bert']['learning_rate']))
        criterion = nn.BCELoss()
        model.train()
        optimizer.zero_grad()

        chunks = preprocessor.divide_into_chunk_pair(impostor_1, impostor_2, chunk_size=config['training']['chunk_size'])

        for chunk_1, chunk_2 in chunks:
            input_ids1, attention_mask1 = preprocessor.tokenize_chunk(chunk_1)
            input_ids2, attention_mask2 = preprocessor.tokenize_chunk(chunk_2)

            similarity = model(input_ids1.to(device), attention_mask1.to(device),
                               input_ids2.to(device), attention_mask2.to(device))

            label = 1 if similarity.item() > 0.5 else 0
        trained_models.append(model.state_dict())

    print("[INFO] Loading Shakespeare data for testing...")
    tested_collection_texts = DataLoader(config['data']['processed_tested_path'])
    tested_collection_data = tested_collection_texts.load_cleaned_text()

    anomaly_scores = []

    for text_idx, text in enumerate(tested_collection_data):
        print(f"[INFO] Processing text {text_idx+1}/{len(tested_collection_data)} from tested collection...")
        batch_size = config['training']['batch_size']
        chunks = preprocessor.divide_into_chunk(text, chunk_size=config['training']['chunk_size'])

        num_batches = len(chunks) // batch_size
        print("[INFO] Number of batches:", num_batches)

        chunks = chunks[:num_batches * batch_size]  # Trim overflow
        print("[INFO] Number of chunks after trimming:", len(chunks))

        rows, cols = num_batches, batch_size
        print("[INFO] Number of rows and columns:", rows, cols)

        signal_representations = []
        dtw_matrix = [[0] * len(trained_models) for _ in range(len(trained_models))]

        for model_idx, model_state in enumerate(trained_models):
            print(f"[INFO] Using trained model {model_idx+1}/{len(trained_models)} for signal extraction...")
            labels_matrix = [[0] * cols for _ in range(rows)]
            model = BertSiameseNetwork().to(device)
            model.load_state_dict(model_state)
            model.eval()

            for i, chunk in enumerate(chunks):
                input_ids, attention_mask = preprocessor.tokenize_chunk(chunk)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                similarity = model.forward_single(input_ids, attention_mask)
                labels_matrix[i // cols][i % cols] = similarity.item()

            signal = [sum(row) / len(row) for row in labels_matrix]
            signal_representations.append(signal)

        print("[INFO] Computing DTW matrix...")
        for i in range(len(signal_representations)):
            for j in range(i + 1, len(signal_representations)):
                dtw_distance = compute_dtw_distance(signal_representations[i], signal_representations[j])
                dtw_matrix[i][j] = dtw_distance

        print("[INFO] Running anomaly detection...")
        anomaly_detector = AnomalyDetector()
        anomaly_mask = anomaly_detector.fit_predict(np.array(dtw_matrix).reshape(-1, 1))
        anomaly_scores.append(anomaly_mask)

    print("[INFO] Running clustering on anomaly scores...")
    clusters = perform_kmedoids_clustering(anomaly_scores, num_clusters=2)

    print("[INFO] Plotting results...")

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title("Clusters of Anomaly Scores")
    plt.xlabel("Cluster")
    plt.ylabel("Anomaly Score")
    plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=clusters, cmap='viridis')
    plt.show()

if __name__ == "__main__":
    full_procedure()
