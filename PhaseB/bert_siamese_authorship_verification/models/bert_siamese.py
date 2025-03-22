import torch
import torch.nn as nn
from transformers import BertModel
import yaml

# Load config values
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class BertSiameseNetwork(nn.Module):
    def __init__(self, bert_model_name=config['model']['bert_model'], hidden_size=config['model']['hidden_size'], dropout=config['model']['dropout']):
        super(BertSiameseNetwork, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)

        # CNN-BiLSTM layers
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):

        # BERT embeddings
        output1 = self.bert(input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.bert(input_ids2, attention_mask=attention_mask2).last_hidden_state

        # CNN layer
        output1 = self.cnn(output1.permute(0, 2, 1))  # (batch, 768, seq_len) -> (batch, 256, seq_len)
        output2 = self.cnn(output2.permute(0, 2, 1))

        # BiLSTM layer
        output1, _ = self.bilstm(output1.permute(0, 2, 1))  # (batch, seq_len, 256) -> (batch, seq_len, 256)
        output2, _ = self.bilstm(output2.permute(0, 2, 1))

        # Pooling (take the last hidden state)
        output1 = output1[:, -1, :]
        output2 = output2[:, -1, :]

        # Compute similarity (Euclidean distance)
        distance = torch.abs(output1 - output2)
        output = self.fc(distance)

        return self.sigmoid(output)

# Model testing
if __name__ == "__main__":
    model = BertSiameseNetwork()
    input_ids = torch.randint(0, 30522, (2, config['model']['max_length']))  # Dummy input (batch_size=2, seq_len=512)
    attention_mask = torch.ones((2, config['model']['max_length']))
    output = model(input_ids, attention_mask, input_ids, attention_mask)
    print(output)