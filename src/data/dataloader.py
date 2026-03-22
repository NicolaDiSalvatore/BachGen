import torch
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    flattened = padded.reshape(padded.size(0), -1)
    return flattened, torch.tensor(lengths)