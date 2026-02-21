import pytest
import torch
from torch import Tensor

from src.data.dataloader import collate_fn


@pytest.fixture
def generate_batch():
    return [(
                torch.tensor([60, 64, 67, 72, 60, 64, 67, 72]),
                8
            ),
            (
                torch.tensor([62, 65, 69, 74]),
                4
            )
    ]

# x, lengths = collate_fn(batch)
# print(x)
def test_collate_fn(generate_batch):
    sequences, lengths = collate_fn(generate_batch)

    # shapes
    assert isinstance(sequences, Tensor)
    assert isinstance(lengths, Tensor)
    assert sequences.shape == (2, 2 * 4)
    assert lengths.tolist() == [8,4]


    # padding correctness
    pad = sequences[1:2, 4:8]
    assert torch.equal(pad, torch.zeros(pad.shape))
