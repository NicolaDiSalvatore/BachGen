import torch
import pytest
from src.models.vae import SequenceVAE 


@pytest.fixture
def model_and_data():
    """
    Returns a fresh VAE model and dummy input data for testing.

    - input_dim: Dimensionality of input vectors
    - hidden_dim: Size of hidden layers
    - latent_dim: Size of the latent vector z
    - seq_len: Number of time steps in the sequence
    - batch_size: Number of samples in a batch
    """
    input_dim = 32
    hidden_dim = 64
    latent_dim = 16
    seq_len = 10
    batch_size = 4

    model = SequenceVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    return model, dummy_input



# Test 1: Check model output shapes
def test_forward_output_shapes(model_and_data):
    """
    Ensure the forward pass produces correct output shapes:
    - recon_x should match input shape
    - mean and logvar should match latent dimension
    """
    model, x = model_and_data
    recon_x, mean, logvar = model(x)

    assert recon_x.shape == x.shape, "Mismatch: recon_x should match input shape"
    assert mean.shape == (x.shape[0], model.latent_dim), "Mean shape incorrect"
    assert logvar.shape == (x.shape[0], model.latent_dim), "Logvar shape incorrect"



# Test 2: Validate loss computation
def test_loss_computation(model_and_data):
    """
    Ensure loss function computes:
    - a positive reconstruction loss
    - a non-negative KL divergence
    - a total loss that is valid
    """
    model, x = model_and_data
    recon_x, mean, logvar = model(x)
    total_loss, recon_loss, kl_loss = model.loss_function(recon_x, x, mean, logvar)

    assert total_loss.item() > 0, "Total loss must be positive"
    assert recon_loss.item() > 0, "Reconstruction loss must be positive"
    assert kl_loss.item() >= 0, "KL divergence must be non-negative"


# Test 3: Ensure gradients flow
def test_backprop(model_and_data):
    """
    Verify that gradients are computed correctly via backpropagation.
    """
    model, x = model_and_data
    recon_x, mean, logvar = model(x)
    loss, _, _ = model.loss_function(recon_x, x, mean, logvar)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in grads), "Gradient contains NaN or Inf"


# Test 4: Model handles all-zero input
def test_zero_input_handling():
    """
    Pass a tensor of zeros and check if the model produces valid output and loss.
    """
    input_dim = 32
    hidden_dim = 64
    latent_dim = 16
    seq_len = 10
    batch_size = 4

    model = SequenceVAE(input_dim, latent_dim, hidden_dim)
    x = torch.zeros(batch_size, seq_len, input_dim)

    recon_x, mean, logvar = model(x)
    assert recon_x.shape == x.shape, "Output shape mismatch for zero input"

    loss, _, _ = model.loss_function(recon_x, x, mean, logvar)
    assert torch.isfinite(loss), "Loss is not finite for zero input"



# Test 5: Model handles large input values
def test_large_input_handling():
    """
    Ensure model behaves correctly with large input values (e.g., scaled data).
    """
    input_dim = 32
    hidden_dim = 64
    latent_dim = 16
    seq_len = 10
    batch_size = 4

    model = SequenceVAE(input_dim, latent_dim, hidden_dim)
    x = torch.randn(batch_size, seq_len, input_dim) * 1e3

    recon_x, mean, logvar = model(x)
    assert torch.isfinite(recon_x).all(), "Reconstruction contains NaN or Inf"

    loss, _, _ = model.loss_function(recon_x, x, mean, logvar)
    assert torch.isfinite(loss), "Loss is NaN/Inf on large input"
