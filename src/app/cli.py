from __future__ import annotations

from app.config import TrainingConfig
from app.training import run_training_pipeline


def main() -> None:
    run_training_pipeline(TrainingConfig())


if __name__ == "__main__":
    main()

