import math

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        limit = math.sqrt(6 / (input_dim + output_dim))
        self.weights = Parameter(torch.empty(output_dim, input_dim).uniform_(-limit, limit))
        # self.weights = Parameter(torch.randn(self.output_dim, self.input_dim))

        if bias:
            self.bias = Parameter(torch.zeros(1, self.output_dim))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        output = torch.matmul(input, self.weights.t())
        if self.bias is not None:
            return output + self.bias
        else:
            return output


