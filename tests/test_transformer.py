import torch
import pytest
import math
from src.models.transformer import PositionalEncoding, MusicTransformer


# Pytest Fixtures and Tests
@pytest.fixture
def model():
    """
    Fixture that returns a MusicTransformer instance for testing.
    """
    return MusicTransformer(input_dim=128, model_dim=64, n_heads=4, n_layers=2, output_dim=128)

def test_output_shape(model):
    """
    Test that the model outputs the correct shape given a dummy input.
    """
    dummy_input = torch.randint(0, 128, (8, 16))  # (batch_size=8, sequence_length=16)
    output = model(dummy_input)
    assert output.shape == (8, 16, 128)


def test_positional_encoding_addition(model):
    """
    Test that positional encoding modifies embeddings but preserves shape and scale.
    """
    dummy_input = torch.randint(0, model.embedding.num_embeddings, (4, 10))  # [batch, seq_len]
    embeddings = model.embedding(dummy_input) * math.sqrt(model.model_dim)

    pos_encoded = model.pos_encoder(embeddings)

    # Shape should be preserved
    assert pos_encoded.shape == embeddings.shape

    # Positional encoding should modify values
    assert not torch.allclose(embeddings, pos_encoded)

    # The modification should not be extreme
    diff = (pos_encoded - embeddings).abs().mean()
    assert diff < 1.0

def test_causal_mask_shape(model):
    """
    Test the causal mask shape and values:
    - 0s on and below diagonal
    - -inf above the diagonal
    """
    seq_len = 16
    mask = model.generate_causal_mask(seq_len)

    assert mask.shape == (seq_len, seq_len)

    # Diagonal should be 0
    assert torch.allclose(mask.diag(), torch.zeros(seq_len), atol=1e-6)

    # Below diagonal should be 0
    below_diag = torch.tril(mask, -1)
    assert torch.allclose(below_diag, torch.zeros_like(below_diag), atol=1e-6)

    # Above diagonal should be -inf
    above_diag = torch.triu(mask, 1)
    assert torch.all(above_diag[above_diag != 0] == float('-inf'))

def test_forward_pass_runs(model):
    """
    Ensure forward pass runs and produces finite output values.
    """
    dummy_input = torch.randint(0, 128, (8, 16))
    output = model(dummy_input)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()