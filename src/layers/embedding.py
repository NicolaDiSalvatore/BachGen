import math

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_matrix = Parameter(
            torch.empty(vocab_size, d_model)
        )
        torch.nn.init.uniform_(self.embedding_matrix, -math.sqrt(1/d_model), math.sqrt(1/d_model))

    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.long
        assert torch.isfinite(x).all()
        assert x.min() >= 0
        assert x.max() < self.vocab_size
        out = self.embedding_matrix[x]*math.sqrt(self.d_model)

        assert torch.isfinite(out).all(), "NaNs created by embedding lookup"
        return out
