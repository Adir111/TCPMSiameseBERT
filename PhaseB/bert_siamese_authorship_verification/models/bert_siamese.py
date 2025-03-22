import torch
import torch.nn as nn
from transformers import BertModel
import yaml

# Load config
# with open('../config/config.yaml', 'r') as f:  # used for training
with open('config/config.yaml', 'r') as f:  # used for main
    config = yaml.safe_load(f)


class BertSiameseNetwork(nn.Module):
    def __init__(self):
        super(BertSiameseNetwork, self).__init__()

        self.bert = BertModel.from_pretrained(config['model']['bert_model'])

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=config['model']['hidden_size'], out_channels=config['model']['cnn_filters'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.bilstm = nn.LSTM(
            input_size=config['model']['cnn_filters'],
            hidden_size=config['model']['lstm_hidden_size'],
            num_layers=config['model']['lstm_num_layers'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(config['model']['lstm_hidden_size'] * 2, 128),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bert(input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.bert(input_ids2, attention_mask=attention_mask2).last_hidden_state

        output1 = self.cnn(output1.permute(0, 2, 1))
        output2 = self.cnn(output2.permute(0, 2, 1))

        output1, _ = self.bilstm(output1.permute(0, 2, 1))
        output2, _ = self.bilstm(output2.permute(0, 2, 1))

        output1 = output1[:, -1, :]
        output2 = output2[:, -1, :]

        distance = torch.abs(output1 - output2)
        return self.fc(distance)


if __name__ == "__main__":
    model = BertSiameseNetwork()
    input_ids = torch.randint(0, 30522, (2, config['model']['max_length']))  # Dummy input (batch_size=2, seq_len=512)
    attention_mask = torch.ones((2, config['model']['max_length']))
    output = model(input_ids, attention_mask, input_ids, attention_mask)
    print(output)
