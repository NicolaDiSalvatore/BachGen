import torch
import pytest
import math
from src.models.transformer import MusicTransformer


@pytest.fixture
def model():
    """
    Fixture that returns a MusicTransformer instance for testing.
    """
    return MusicTransformer(input_dim=4, model_dim=64, n_heads=4, n_layers=2, output_dim=4)


def test_output_shape(model):
    """
    Test that the model outputs the correct shape given a dummy input.
    """
    dummy_input = torch.randn(8, 16, 4)  # (batch_size=8, seq_len=16, input_dim=4)
    lengths = torch.tensor([16] * 8)
    output = model(dummy_input, lengths=lengths)
    assert output.shape == (8, 16, 4)


def test_positional_encoding_addition(model):
    """
    Test that positional encoding modifies embeddings but preserves shape and does not explode.
    """
    dummy_embeddings = torch.randn(4, 10, model.model_dim)  # Simulated input embeddings
    pos_encoded = model.pos_encoder(dummy_embeddings)

    # Shape should be preserved
    assert pos_encoded.shape == dummy_embeddings.shape

    # Positional encoding should modify values
    assert not torch.allclose(dummy_embeddings, pos_encoded)

    # The modification should not be extreme (adjust threshold if needed)
    diff = (pos_encoded - dummy_embeddings).abs().mean()
    assert diff < 1.5, f"Positional encoding modified values too much (mean diff: {diff:.4f})"


def test_generate_causal_mask_shape_and_values(model):
    seq_len = 16
    mask = model.generate_causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)
    # Diagonal 0
    assert torch.allclose(mask.diag(), torch.zeros(seq_len), atol=1e-6)
    # Below diagonal 0
    below_diag = torch.tril(mask, -1)
    assert torch.allclose(below_diag, torch.zeros_like(below_diag), atol=1e-6)
    # Above diagonal -inf
    above_diag = torch.triu(mask, 1)
    assert torch.all(above_diag[above_diag != 0] == float('-inf'))

def test_forward_mask_shape_and_dtype(model):
    batch_size = 8
    seq_len = 16
    input_dim = model.embedding.in_features

    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Random lengths with some padding
    lengths = torch.randint(low=5, high=seq_len+1, size=(batch_size,))

    # Run forward to extract the mask shape indirectly (by catching runtime errors)
    output = model(dummy_input, lengths=lengths)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model.fc_out.out_features)


def test_forward_pass_runs(model):
    """
    Ensure forward pass runs and produces finite output values.
    """
    dummy_input = torch.randn(8, 16, 4)
    lengths = torch.tensor([12, 10, 16, 14, 15, 16, 13, 11])
    output = model(dummy_input, lengths=lengths)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()