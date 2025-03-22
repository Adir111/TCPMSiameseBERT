import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from bert_siamese_authorship_verification.models.bert_siamese import BertSiameseNetwork
from data_loader import DataLoader

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertSiameseNetwork().to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
criterion = nn.BCELoss()

# Load data
data_loader = DataLoader(config['data']['train_path'])
train_data = data_loader.load_data()


# Training loop
def train():
    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        for text1, text2, label in train_data:
            input_ids1, attention_mask1 = data_loader.tokenize_pairs(text1, text2)

            input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)
            labels = torch.tensor([label], dtype=torch.float).to(device)

            optimizer.zero_grad()
            output = model(input_ids1, attention_mask1, input_ids1, attention_mask1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_data)}")


if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "models/bert_siamese.pth")
