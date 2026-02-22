import torch
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input: each batch is a list of tuples (sequence, number of time steps); the sequence is a tensor with shape (number of time steps *4)
    # Output: the sequences are padded to have the same number of time steps. The output tensor with shape (batch size, max number of time steps*4)

    # Purpose: Pads sequences (of variable length) inside the batch so they can be stacked into a single tensor
    sequences, lengths = zip(*batch)

    return pad_sequence(sequences, batch_first=True), torch.tensor(lengths)