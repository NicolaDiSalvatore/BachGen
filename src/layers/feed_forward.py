"""
Summary:

1. Input Preparation:
   - Receive input tensor of token IDs (shape: batch_size x seq_len).
   - Convert tokens to embeddings of size model_dim.
   - Scale embeddings by sqrt(model_dim) to normalize variance.

2. Inject Positional Information:
   - Generate fixed sinusoidal positional encodings.
   - Add positional encodings to scaled embeddings.
   - Apply dropout for regularization.

3. Create Causal Mask:
   - Generate square mask to prevent attention to future tokens.
   - Ensures autoregressive property: token at position i attends only to tokens ≤ i.

4. Transformer Encoding:
   - Pass embeddings + positional encodings through multi-layer Transformer encoder.
   - Each encoder layer applies:
     a) Multi-head self-attention with causal masking.
     b) Feedforward neural network with activation.
     c) Residual connections and layer normalization.
   - Captures contextual dependencies across sequence positions.

5. Output Projection:
   - Apply linear layer to encoder output (shape: batch_size x seq_len x model_dim).
   - Project to output_dim (vocabulary size) logits for token prediction.

6. Output:
   - Return logits tensor with shape (batch_size, seq_len, output_dim).
   - Use for next-token prediction during training or autoregressive sampling.

"""

import torch
from torch import Tensor
from torch.nn import Module, Sequential

from src.layers.linear import Linear
from src.layers.relu import Relu


def relu(x: Tensor) -> Tensor:
    return torch.max(torch.zeros(x.shape), x)


class FeedForward(Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.feed_forward = Sequential(
            Linear(self.input_dim, self.hidden_dim),
            Relu(),
            Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)
