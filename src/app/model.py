from __future__ import annotations

import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "relu":
        return nn.ReLU()
    if normalized_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if normalized_name == "gelu":
        return nn.GELU()
    if normalized_name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'.")


class AdultIncomeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_build_activation(activation))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
