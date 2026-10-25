"""Small trainable classifier; pose detector parameters stay frozen."""

from typing import Sequence

import torch
from torch import nn


class PostureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        num_classes: int = 4,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("input_dim and num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        layers = []
        previous = input_dim
        for size in hidden_dims:
            layers.extend((nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)))
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
