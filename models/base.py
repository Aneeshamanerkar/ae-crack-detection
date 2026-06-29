# models/base.py
import torch.nn as nn
from abc import ABC, abstractmethod

class SequenceLabeler(nn.Module, ABC):
    """
    Abstract Base Class for AE models.
    All models must adhere to the contract:
    Input:  (Batch, Time, Features/Freq)
    Output: (Batch, Time, Classes)
    """
    def __init__(self, n_freq, n_classes):
        super().__init__()
        self.n_freq = n_freq
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B, T, F)
        Returns:
            logits: torch.Tensor of shape (B, T, C)
        """
        pass