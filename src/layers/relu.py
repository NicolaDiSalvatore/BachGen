import torch
from torch import Tensor
from torch.nn import Module


class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.maximum(torch.zeros_like(x), x)
