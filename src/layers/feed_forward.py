
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
