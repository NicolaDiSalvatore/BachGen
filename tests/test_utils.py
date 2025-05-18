import os
import torch
import yaml
import random
import numpy as np

from src.utils import set_seed, save_checkpoint, load_config


def test_set_seed():
    # Set seed and generate random numbers
    set_seed(123)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    # Reset seed and generate again, should be the same
    set_seed(123)
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2, "Random seed does not produce consistent results"
    assert n1 == n2, "Numpy seed does not produce consistent results"
    assert t1 == t2, "Torch seed does not produce consistent results"


def test_save_checkpoint_and_load(tmp_path):
    # Setup dummy model and optimizer
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())

    epoch = 5
    val_loss = 0.123
    checkpoint_path = tmp_path / "checkpoint.pt"

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

    # Load checkpoint file back
    checkpoint = torch.load(checkpoint_path)

    assert checkpoint['epoch'] == epoch
    assert isinstance(checkpoint['model_state_dict'], dict)
    assert isinstance(checkpoint['optimizer_state_dict'], dict)
    assert checkpoint['val_loss'] == val_loss


def test_load_config(tmp_path):
    # Create dummy yaml config file
    config_dict = {
        'training': {'epochs': 10, 'batch_size': 16},
        'model': {'hidden_size': 128}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

    # Load with function
    loaded_config = load_config(config_path)

    assert loaded_config == config_dict