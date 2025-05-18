import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int):
    """
    Set seed for reproducibility across random, numpy, and torch (CPU and GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """
    Save model and optimizer state along with epoch and validation loss info.
    
    Args:
        model: PyTorch model instance
        optimizer: optimizer instance
        epoch: current epoch number (int)
        val_loss: validation loss (float)
        filepath: Path or str where to save the checkpoint (.pt file)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file and return as dictionary.
    
    Args:
        config_path: Path or str to the YAML config file
    
    Returns:
        Dictionary with config parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config