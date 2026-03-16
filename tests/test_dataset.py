import json
import pytest
import torch

from src.data.dataset import BachDataset


@pytest.fixture(scope="module")
def mock_data_and_save(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    mock_data_path = tmp_dir / "mock_data.json"

    mock_data = {
      "train": [
        [[60, 62, 64, 65], [67, 69, 71, 72]],
        [[55, 57, 59, 60], [50, 52, 53, 54], [55, 56, 58, 59]]
      ],
      "valid": [
        [[45, 47, 49, 50], [52, 54, 55, 57]]
      ],
      "test": [
        [[65, 67, 68, 70], [72, 73, 75, 76]],
        [[40, 42, 43, 45]]
      ]
    }
    with open(mock_data_path, "w") as f:
        json.dump(mock_data, f)

    return mock_data_path


def test_bachdataset(mock_data_and_save):

    data_path = mock_data_and_save
    data = BachDataset("test", data_path)

    print(data)

    seq, length = data[0]

    assert seq.shape == (2, 4)
    assert length == 2
    assert isinstance(seq, torch.Tensor)

    with pytest.raises(ValueError):
        BachDataset("tain", data_path)

