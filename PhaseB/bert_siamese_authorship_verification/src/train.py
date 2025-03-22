import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from bert_siamese_authorship_verification.models.bert_siamese import BertSiameseNetwork
from data_loader import DataLoader
from preprocess import TextPreprocessor
from evaluate import evaluate_model
from dtw import compute_dtw_distance

# Load config
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertSiameseNetwork().to(device)
optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
criterion = nn.BCELoss()

data_loader = DataLoader(config['data']['train_path'])
train_data = data_loader.load_data()
preprocessor = TextPreprocessor()


def train():
    model.train()
    y_true = []
    y_pred = []
    dtw_scores = []

    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for text1, text2, label in train_data:
            input_ids1, attention_mask1 = preprocessor.tokenize_pairs(text1, text2)
            input_ids1, attention_mask1 = (input_ids1[:, :config['model']['max_length']],
                                           attention_mask1[:, :config['model']['max_length']])
            labels = torch.tensor([label], dtype=torch.float).unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(input_ids1.to(device), attention_mask1.to(device), input_ids1.to(device), attention_mask1.to(device))
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


if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "../models/bert_siamese.pth")
