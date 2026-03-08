from torch import Tensor, exp
from torch.nn import Module


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.max(dim=self.dim, keepdim=True).values
        exp_x = exp(x)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)
