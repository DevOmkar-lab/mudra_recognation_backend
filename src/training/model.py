"""Small landmark-vector classifier suitable for real-time CPU inference."""
from __future__ import annotations
import torch
from torch import nn

class MudraClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2: raise ValueError("At least two Mudra classes are required")
        layers: list[nn.Module] = []; previous = input_size
        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]; previous = size
        layers.append(nn.Linear(previous, num_classes)); self.network = nn.Sequential(*layers)
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.network(features)
