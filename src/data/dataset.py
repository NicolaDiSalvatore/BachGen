"""
This script defines the dataset and data loading utilities. It assumes the data has already 
been processed into lists of tensors, each of shape [T_x, 4], and saved as .pt files.

Key Components:
---------------
1. BachDataset (torch.utils.data.Dataset):
   - Wraps a list of chorale sequences (as torch tensors).
   - Each sequence is a 2D tensor of shape [T_x, 4] (T_x = time steps).
   - Supports optional padding/truncation to a fixed `max_seq_len`.

2. load_dataset(path, max_seq_len):
   - Loads a .pt file containing a list of tensors.
   - Returns a `BachDataset` instance.

3. get_dataloader(path, batch_size, shuffle, max_seq_len):
   - Wraps `load_dataset()` and returns a PyTorch `DataLoader`.
   - Includes a `collate_fn` that dynamically pads sequences in each batch to match
     the longest sequence in that batch.
   - Returns batches of shape [batch_size, max_seq_len, 4] and sequence lengths.

"""



import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List


class BachDataset(Dataset):
    """
    Dataset for Bach chorales represented as tensor sequences.
    Each item is a tensor of shape [T_x, 4], where T_x is sequence length.
    """

    def __init__(self, data: List[torch.Tensor]):
        """
        Args:
            data (List[torch.Tensor]): List of sequences (tensors) [T_x, 4].
        """
        for i, seq in enumerate(data):
            if seq.ndim != 2 or seq.size(1) != 4:
                raise ValueError(f"Sequence at index {i} must have shape [T_x, 4], got {seq.shape}")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_dataset(path: str) -> BachDataset:
    """
    Load a processed dataset from .pt file.

    Args:
        path (str): Path to .pt file (list of [T_x, 4] tensors).

    Returns:
        BachDataset: Dataset instance.
    """
    data = torch.load(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected data to be a list of tensors, got {type(data)}")
    return BachDataset(data)


def get_dataloader(path: str, batch_size: int, shuffle: bool = True, max_seq_len: Optional[int] = None) -> DataLoader:
    """
    Create DataLoader for training/validation/testing.

    Args:
        path (str): Path to dataset .pt file.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle dataset.
        max_seq_len (int, optional): Max sequence length for truncation/padding.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = load_dataset(path)

    def collate_fn(batch: List[torch.Tensor]):
        lengths = [seq.size(0) for seq in batch]
        max_len = max_seq_len if max_seq_len is not None else max(lengths)

        padded_batch = []
        for seq in batch:
            seq_len = seq.size(0)
            if seq_len > max_len:
                seq = seq[:max_len]
            elif seq_len < max_len:
                pad = torch.zeros(max_len - seq_len, seq.size(1), dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=0)
            padded_batch.append(seq)

        batch_tensor = torch.stack(padded_batch)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        return batch_tensor, lengths_tensor

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
