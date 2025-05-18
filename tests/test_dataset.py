import pytest
import torch
import os
from src.data.dataset import BachDataset, load_dataset, get_dataloader

@pytest.fixture(scope="module")
def mock_data_and_path():
    mock_data = [
        torch.tensor([[60, 62, 64, 65], [67, 69, 71, 72]]),  
        torch.tensor([[55, 57, 59, 60]]),                    
        torch.tensor([[50, 52, 53, 54],                      
                      [55, 56, 58, 59],
                      [60, 61, 62, 63]])
    ]
    mock_data_path = "mock_data.pt"
    torch.save(mock_data, mock_data_path)

    yield mock_data, mock_data_path

    # Cleanup after tests
    if os.path.exists(mock_data_path):
        os.remove(mock_data_path)


def test_bachdataset_init(mock_data_and_path):
    mock_data, _ = mock_data_and_path
    dataset = BachDataset(mock_data)

    assert len(dataset) == 3
    assert torch.equal(dataset[0], mock_data[0])


def test_load_dataset(mock_data_and_path):
    mock_data, path = mock_data_and_path
    dataset = load_dataset(path)

    assert len(dataset) == 3
    assert torch.equal(dataset[2], mock_data[2])


def test_get_dataloader(mock_data_and_path):
    mock_data, path = mock_data_and_path
    dataloader = get_dataloader(
        path,
        batch_size=2,
        shuffle=False,
        max_seq_len=4
    )

    batch_tensor, lengths_tensor = next(iter(dataloader))

    assert batch_tensor.shape == (2, 4, 4)
    assert lengths_tensor.tolist() == [2, 1]

    assert torch.equal(batch_tensor[0, :2], mock_data[0])
    assert torch.equal(batch_tensor[1, :1], mock_data[1])
    assert torch.equal(batch_tensor[1, 1:], torch.zeros(3, 4))