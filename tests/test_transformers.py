import pytest
import torch

# Import your model
# adjust the import path if needed
from src.models.transformer import Decoder, MusicTransformer


@pytest.fixture
def batch_size():
    return 4



@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def vocab_size():
    return 128


@pytest.fixture
def attention_hidden_dim():
    return 96


@pytest.fixture
def feedforward_hidden_dim():
    return 192

@pytest.fixture
def num_attention_heads():
    return 6





@pytest.fixture
def dummy_input(batch_size, seq_len, vocab_size):
    """
    Fake tokenized music sequence
    """
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def decoder(attention_hidden_dim, feedforward_hidden_dim, seq_len, num_attention_heads):
    return Decoder(attention_hidden_dim=attention_hidden_dim,
                   seq_len=seq_len,
                   feedforward_hidden_dim=feedforward_hidden_dim,
                   num_attention_heads=num_attention_heads,
                   ffn_dropout=0.1,
                   attn_dropout=0.1,
                   attn_proj_dropout=0.1
    )


@pytest.fixture
def model(vocab_size, seq_len, attention_hidden_dim, feedforward_hidden_dim, num_attention_heads):
    return MusicTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_decoder_layers=2,
        attention_hidden_dim=attention_hidden_dim,
        feedforward_hidden_dim=feedforward_hidden_dim,
        num_attention_heads=num_attention_heads,
        embed_dropout=0.1,
        ffn_dropout=0.1,
        attn_dropout=0.1,
        attn_proj_dropout=0.1
    )


# ------------------------
# Decoder tests
# ------------------------

def test_decoder_forward_shape(decoder, batch_size, seq_len, attention_hidden_dim):
    # Use shorter sequence length to test dynamic sizing
    short_len = seq_len // 2
    x = torch.randn(batch_size, short_len, attention_hidden_dim)
    y = decoder(x)

    assert y.shape == (batch_size, short_len, attention_hidden_dim)


def test_decoder_backward(decoder, batch_size, seq_len, attention_hidden_dim):
    x = torch.randn(batch_size, seq_len, attention_hidden_dim, requires_grad=True)
    y = decoder(x)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ------------------------
# MusicTransformer tests
# ------------------------

def test_music_transformer_forward_shape(model, dummy_input, batch_size, seq_len, vocab_size):
    y = model(dummy_input)

    assert y.shape == (batch_size, seq_len, vocab_size)


def test_music_transformer_softmax_properties(model, dummy_input):
    y = model(dummy_input)

    # Model outputs logits, apply softmax
    probs = torch.softmax(y, dim=-1)

    # Probabilities should be >= 0
    assert torch.all(probs >= 0)

    # Softmax over last dim should sum to 1
    probs_sum = probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_music_transformer_backward(model, dummy_input):
    y = model(dummy_input)

    loss = y.mean()
    loss.backward()

    # Check at least one parameter received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_multiple_decoder_layers_consistency(vocab_size, seq_len, attention_hidden_dim, feedforward_hidden_dim, num_attention_heads, dummy_input):
    model_1 = MusicTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_decoder_layers=1,
        attention_hidden_dim=attention_hidden_dim,
        feedforward_hidden_dim=feedforward_hidden_dim,
        num_attention_heads=num_attention_heads
    )

    model_3 = MusicTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_decoder_layers=3,
        attention_hidden_dim=attention_hidden_dim,
        feedforward_hidden_dim=feedforward_hidden_dim,
        num_attention_heads=num_attention_heads
    )

    y1 = model_1(dummy_input)
    y3 = model_3(dummy_input)

    assert y1.shape == y3.shape
