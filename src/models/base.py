import torch
from abc import ABC, abstractmethod

# Step-by-Step Description:
# This script defines a base class for models that ensures consistency across model types.
# The Music Transformer and VAE models will inherit from this base class to maintain a common
# interface for forward pass, loss computation, and training steps.

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        """
        Abstract method for the forward pass of the model.
        """
        pass
    
    @abstractmethod
    def loss_function(self, *args, **kwargs):
        """
        Abstract method to compute the loss.
        """
        pass

    @abstractmethod
    def train_step(self, x):
        """
        Abstract method for a single training step.
        """
        pass