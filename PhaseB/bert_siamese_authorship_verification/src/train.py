import torch
import torch.nn as nn
import torch.optim as optim
from bert_siamese_authorship_verification.models.bert_siamese import BertSiameseNetwork
from bert_siamese_authorship_verification.config.get_config import get_config
from data_loader import DataLoader
from preprocess import TextPreprocessor
from evaluate import evaluate_model
from dtw import compute_dtw_distance
from isolation_forest import AnomalyDetector

# Load config
config = get_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertSiameseNetwork().to(device)

# Define optimizer with BERT learning rate
optimizer = optim.AdamW(
    model.parameters(),
    lr=float(config['bert']['learning_rate'])
)

criterion = nn.BCELoss()

# Load Data
data_loader = DataLoader(config['data']['train_path'])
train_data = data_loader.load_data()
preprocessor = TextPreprocessor()


# Convert text embeddings for anomaly detection
def get_embedding(text1, text2):
    input_ids, attention_mask = preprocessor.tokenize_pairs(text1, text2)
    with torch.no_grad():
        embedding = model.bert(input_ids.to(device), attention_mask.to(device)).pooler_output.cpu().numpy()
    return embedding.flatten()


train_features = [get_embedding(t1, t2) for t1, t2, _ in train_data]
anomaly_detector = AnomalyDetector()
mask = anomaly_detector.fit_predict(train_features)

# Filter out anomalies
filtered_data = [train_data[i] for i in range(len(train_data)) if mask[i]]


def train():
    model.train()
    y_true = []
    y_pred = []
    dtw_scores = []

    best_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    no_improve_epochs = 0

    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for text1, text2, label in train_data:
            input_ids1, attention_mask1 = preprocessor.tokenize_pairs(text1, text2)
            input_ids1, attention_mask1 = (input_ids1[:, :config['bert']['maximum_sequence_length']],
                                           attention_mask1[:, :config['bert']['maximum_sequence_length']])
            labels = torch.tensor([label], dtype=torch.float).unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(input_ids1.to(device), attention_mask1.to(device), input_ids1.to(device),
                           attention_mask1.to(device))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Compute DTW score
            dtw_distance = compute_dtw_distance(output.cpu().detach().numpy(), labels.cpu().detach().numpy())
            dtw_scores.append(dtw_distance)
            y_true.append(label)
            y_pred.append(1 if output.item() > 0.5 else 0)

        metrics = evaluate_model(y_true, y_pred)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_data)}, Metrics: {metrics}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "../models/bert_siamese.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered. Training stopped.")
                break


if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "../models/bert_siamese.pth")
