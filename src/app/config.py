from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    hidden_dim_3: int = 64
    learning_rate: float = 1e-3
    epochs: int = 25
    patience: int = 5
    dropout: float = 0.2
    random_state: int = 42
