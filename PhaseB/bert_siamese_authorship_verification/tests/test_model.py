import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_siamese import BertSiameseNetwork
from config.get_config import get_config


def test_model_forward_shapes():
    config = get_config()
    model = BertSiameseNetwork()
    batch_size = 2
    seq_len = config['bert']['maximum_sequence_length']

    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    output = model(input_ids, attention_mask, input_ids, attention_mask)
    assert output.shape == (batch_size, 1), "Output shape should be [batch_size, 1]"
    assert torch.isfinite(output).all(), "Model output contains NaNs or Infs"


def test_forward_single_embedding_shape():
    config = get_config()
    model = BertSiameseNetwork()
    seq_len = config['bert']['maximum_sequence_length']
    input_ids = torch.randint(0, 30522, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))

    emb = model.forward_single(input_ids, attention_mask)
    assert emb.shape == (1, 1), "forward_single should return [1, 1] shape"
    assert torch.isfinite(emb).all(), "forward_single contains invalid values"


def test_model_device_assignment():
    model = BertSiameseNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.parameters():
        assert param.device == device, f"Model parameter on {param.device}, expected {device}"


def test_model_parameter_count():
    model = BertSiameseNetwork()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 1e6, f"Expected >1M parameters, got {num_params}"


def test_consistency_between_calls():
    config = get_config()
    model = BertSiameseNetwork()
    input_ids = torch.randint(0, 30522, (1, config['bert']['maximum_sequence_length']))
    attention_mask = torch.ones((1, config['bert']['maximum_sequence_length']))

    output1 = model.forward_single(input_ids, attention_mask)
    output2 = model.forward_single(input_ids, attention_mask)

    diff = (output1 - output2).abs().max().item()
    assert diff < 1e-4, "Outputs from identical inputs should be consistent (no dropout during eval)"
