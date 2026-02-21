import pytest
import torch

from src.data.tokenizer import BachTokenizer


@pytest.fixture(scope="module")
def mock_data():
    mock_input_data = {
        "train": torch.tensor([
            [[60, 62, 64, 65], [67, 69, 71, 72]],
            [[55, 57, 59, 60], [50, 52, 53, 54]]
        ]),
        "valid": torch.tensor([
            [[45, 47, 49, 50], [52, 54, 55, 57]]
        ]),
        "test": torch.tensor([
            [[65, 67, 68, 70], [72, 73, 75, 76]],
            [[40, 42, 43, 45], [55, 56, 58, 59]]
        ])
    }

    mock_output_data = {
        "train": torch.tensor([
            [60, 62, 64, 65, 67, 69, 71, 72],
            [55, 57, 59, 60, 50, 52, 53, 54]
        ]),
        "valid": torch.tensor([
            [45, 47, 49, 50, 52, 54, 55, 57]
        ]),
        "test": torch.tensor([
            [65, 67, 68, 70, 72, 73, 75, 76],
            [40, 42, 43, 45, 55, 56, 58, 59]
        ])
    }

    return mock_input_data, mock_output_data

def test_tokenizer(mock_data):

    input_data, output_data = mock_data

    tokenizer = BachTokenizer()
    for split in ["train", "valid", "test"]:
        tokenized_data = tokenizer.encode(input_data[split])
        assert torch.equal(tokenized_data, output_data[split])



