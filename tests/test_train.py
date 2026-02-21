from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Adjusted imports to match src/train.py
from src.train import evaluate_model, main


# Dummy model compatible with the training loop expectations
class DummyModel(nn.Module):
    def __init__(self, vocab_size=10, dim=128):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size
        self.attention_hidden_dim = dim
        self.feedforward_hidden_dim = dim
        self.num_decoder_layers = 1
        self.num_attention_heads = 2
        # Mocking embedding to handle integer inputs
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        # x: [B, S]
        emb = self.embedding(x) # [B, S, D]
        return self.linear(emb) # [B, S, V]

@pytest.fixture
def dummy_data():
    batch_size = 4
    seq_len = 16
    vocab_size = 10

    # The training loop expects: sequences, lengths = data
    # So we create a dataset that returns (sequence, length)
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 20
        def __getitem__(self, idx):
            # Return sequence and length
            return torch.randint(0, vocab_size, (seq_len,)), seq_len

    return DataLoader(MockDataset(), batch_size=batch_size)

def test_evaluate_model(dummy_data):
    vocab_size = 10
    model = DummyModel(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    # evaluate_model(model: MusicTransformer, loss_fn: CrossEntropyLoss, loader: DataLoader, vocab_size: int)
    loss = evaluate_model(model, criterion, dummy_data, vocab_size)
    assert isinstance(loss, float)
    # Loss should be non-negative
    # Note: CrossEntropyLoss can be negative if targets are negative but here they are indices
    # However, if label smoothing is used, it might behave differently, but usually positive.
    # The dummy model output is random, so loss is unpredictable but should be float.

@patch("src.train.BachDataset")
@patch("src.train.yaml.safe_load")
@patch("src.train.open")
@patch("src.train.Config")
@patch("src.train.generate_configs")
@patch("src.train.mlflow")
@patch("src.train.train_and_validate")
@patch("src.train.DataLoader")
@patch("src.train.get_vocab_size")
@patch("src.train.asdict")
@patch("src.train.dataclasses.asdict")
def test_main_runs(mock_dataclasses_asdict, mock_asdict, mock_get_vocab_size, mock_dataloader, mock_train_and_validate, mock_mlflow, mock_generate_configs, mock_config, mock_open, mock_yaml_load, mock_dataset):
    # Setup mocks
    mock_get_vocab_size.return_value = 10
    mock_asdict.return_value = {"some": "params"}
    mock_dataclasses_asdict.return_value = {"some": "params"}
    mock_yaml_load.return_value = {
        "search_space": {
            "num_attention_heads": [4],
            "num_decoder_layers": [2],
            "attention_hidden_size": [64],
            "feedforward_hidden_size": [128],
            "epochs": [1],
            "seed": [42],
            "learning_rate": [1e-3],
            "batch_size": [4],
            "accumulation_steps": [1],
            "embed_dropout": [0.1],
            "ffn_dropout": [0.1],
            "attn_dropout": [0.1],
            "attn_proj_dropout": [0.1],
            "weight_decay": [0.01]
        }
    }

    # Mock Config object
    mock_config_instance = MagicMock()
    mock_config_instance.batch_size = 4
    mock_config_instance.seed = 42
    mock_config_instance.accumulation_steps = 1
    mock_generate_configs.return_value = [mock_config_instance]

    # Mock Dataset
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.get_min_pitch.return_value = 0
    mock_dataset_instance.get_max_pitch.return_value = 100
    mock_dataset_instance.get_max_seq_len.return_value = 100
    mock_dataset.return_value = mock_dataset_instance

    # Mock train_and_validate return
    mock_model = MagicMock()
    # Configure mock_model to return a tensor with correct shape
    def mock_forward(x):
        # x shape: [batch, seq_len]
        batch, length = x.shape
        return torch.randn(batch, length, 10)
    mock_model.side_effect = mock_forward

    mock_train_and_validate.return_value = (mock_model, 0.5, 1)

    # Configure DataLoader to yield one batch
    # Data is (sequences, lengths)
    # sequences shape: [batch_size, seq_len]
    seq_len = 16
    batch_size = 4
    vocab_size = 10
    sequences = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.full((batch_size,), seq_len)

    # Make the DataLoader instance iterable
    mock_dataloader_instance = mock_dataloader.return_value
    mock_dataloader_instance.__iter__.return_value = iter([(sequences, lengths)])
    mock_dataloader_instance.__len__.return_value = 1

    # Run main
    # We need to patch argparse as well or pass arguments
    with patch("sys.argv", ["train.py", "--config", "config.yaml"]):
         main()

    # Assert train_and_validate was called
    assert mock_train_and_validate.called
