import torch

class BachTokenizer:
    def __init__(self):
        pass
    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.dim() != 3:
            raise ValueError("Wrong dimensions for sequences")
        return torch.flatten(sequences, start_dim = 1)






