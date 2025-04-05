import torch
import torch.nn as nn
from transformers import BertModel
from config.get_config import get_config

# Load config
config = get_config()


class BertSiameseNetwork(nn.Module):
    def __init__(self):
        super(BertSiameseNetwork, self).__init__()

        # Load BERT model
        self.bert = BertModel.from_pretrained(config['bert']['model'])

        # CNN-BiLSTM Block
        self.conv = nn.Conv1d(
            in_channels=config['model']['hidden_size'],
            out_channels=config['model']['cnn']['filters'],
            kernel_size=config['model']['cnn']['kernel_size'],
            stride=config['model']['cnn']['stride'],
            padding=config['model']['cnn']['padding']
        )

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.bilstm = nn.LSTM(
            input_size=config['model']['cnn']['filters'],
            hidden_size=config['model']['bilstm']['hidden_units'],
            num_layers=config['model']['bilstm']['number_of_layers'],
            bidirectional=config['model']['bidirectional'],  # True = BiLSTM
            batch_first=True
        )

        self.dropout = nn.Dropout(config['model']['bilstm']['dropout'])

        self.fc_relu = nn.Sequential(
            nn.Linear(config['model']['bilstm']['hidden_units'] * 2, 128),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=config['model']['softmax_dim'])

        # Final FC + ReLU after CNN-BiLSTM
        self.final_fc_relu = nn.Sequential(
            nn.Linear(config['model']['fc']['in_features'], config['model']['fc']['out_features']),
            nn.ReLU(),
            nn.Linear(config['model']['fc']['out_features'], 1)
        )


    @staticmethod
    def mean_pooling(bert_output, attention_mask):
        """Compute mean pooling for BERT embeddings"""
        token_embeddings = bert_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / sum_mask

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # BERT Processing
        bert_output1 = self.bert(input_ids1, attention_mask=attention_mask1)
        bert_output2 = self.bert(input_ids2, attention_mask=attention_mask2)

        # Mean Pooling
        pooled_output1 = self.mean_pooling(bert_output1, attention_mask1)
        pooled_output2 = self.mean_pooling(bert_output2, attention_mask2)

        # CNN expects (batch_size, channels, seq_len)
        # Reshape to (batch_size, hidden_size, 1)
        pooled_output1 = pooled_output1.unsqueeze(2)  # (batch_size, 1, hidden_size)
        pooled_output2 = pooled_output2.unsqueeze(2)  # (batch_size, 1, hidden_size)

        # CNN Processing
        conv_out1 = self.conv(pooled_output1)  # (batch_size, filters, seq_len)
        conv_out2 = self.conv(pooled_output2)  # (batch_size, filters, seq_len)

        # Max Pooling
        pool_out1 = self.max_pool(conv_out1)  # (batch_size, filters, 1)
        pool_out2 = self.max_pool(conv_out2)  # (batch_size, filters, 1)

        # The BiLSTM expects input with shape (batch_size, seq_len, input_size)
        bilstm_out1, _ = self.bilstm(pool_out1.permute(0, 2, 1))  # (batch_size, seq_len, hidden_size)
        bilstm_out2, _ = self.bilstm(pool_out2.permute(0, 2, 1))  # (batch_size, seq_len, hidden_size)

        # Extract last hidden state
        bilstm_out1 = bilstm_out1[:, -1, :]  # (batch_size, hidden_size)
        bilstm_out2 = bilstm_out2[:, -1, :]  # (batch_size, hidden_size)

        # Dropout
        bilstm_out1 = self.dropout(bilstm_out1)
        bilstm_out2 = self.dropout(bilstm_out2)

        # FC + ReLU
        fc_relu_out1 = self.fc_relu(bilstm_out1)
        fc_relu_out2 = self.fc_relu(bilstm_out2)

        # Softmax
        softmax_out1 = self.softmax(fc_relu_out1)
        softmax_out2 = self.softmax(fc_relu_out2)

        # Final FC + ReLU
        final_out1 = self.final_fc_relu(softmax_out1)
        final_out2 = self.final_fc_relu(softmax_out2)

        # Euclidean distance between vectors (batch-wise)
        distance = torch.abs(final_out1 - final_out2)  # shape: [batch_size, 1]
        return distance

    def forward_single(self, input_ids, attention_mask):
        # no gradient because forward_single is meant to be used to process a single chunk, so
        # it's used when we're processing the tested collection - post training.
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = self.mean_pooling(bert_output, attention_mask).unsqueeze(2)
            conv_out = self.conv(pooled_output)
            pool_out = self.max_pool(conv_out)
            bilstm_out, _ = self.bilstm(pool_out.permute(0, 2, 1))
            bilstm_out = self.dropout(bilstm_out[:, -1, :])
            fc_out = self.fc_relu(bilstm_out)
            softmax_out = self.softmax(fc_out)
            final_score = self.final_fc_relu(softmax_out)
            return final_score


if __name__ == "__main__":
    model = BertSiameseNetwork()

    # Create dummy input
    batch_size = 2
    seq_length = config['bert']['maximum_sequence_length']
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    _attention_mask = torch.ones((batch_size, seq_length))

    # Test forward pass
    output = model(input_ids, _attention_mask, input_ids, _attention_mask)
    print(output)
