from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str | None = None
    seed: int = 42
    batch_size: int = 256
    epochs: int = 25
    patience: int = 5
    threshold: float = 0.5
    final_evaluation: bool = False
    artifact_root: str = "artifacts"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @property
    def random_state(self) -> int:
        return self.seed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
