from __future__ import annotations

import time
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import ModelConfig, OptimizerConfig, TrainingConfig
from app.data import load_preprocessed_adult_data
from app.model import AdultIncomeMLP
from app.training import initialize_run_artifacts, update_leaderboard, write_json


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainingConfig) -> tuple[dict[str, DataLoader], float, int]:
    prepared = load_preprocessed_adult_data(random_state=config.seed)

    train_dataset = TensorDataset(
        torch.from_numpy(prepared.X_train),
        torch.from_numpy(prepared.y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(prepared.X_val),
        torch.from_numpy(prepared.y_val),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(prepared.X_test),
        torch.from_numpy(prepared.y_test),
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    }

    pos_weight = prepared.class_weights[1] / prepared.class_weights[0]
    input_dim = prepared.X_train.shape[1]
    return dataloaders, pos_weight, input_dim


def compute_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    targets = targets.float()

    correct = (predictions == targets).sum().item()
    accuracy = correct / max(len(targets), 1)

    true_positive = ((predictions == 1) & (targets == 1)).sum().item()
    true_negative = ((predictions == 0) & (targets == 0)).sum().item()
    false_positive = ((predictions == 1) & (targets == 0)).sum().item()
    false_negative = ((predictions == 0) & (targets == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        loss = criterion(logits, targets)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_logits = torch.cat(all_logits)
    epoch_targets = torch.cat(all_targets)
    metrics = compute_binary_metrics(epoch_logits, epoch_targets, threshold=threshold)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    threshold: float,
) -> dict[str, Any]:
    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            threshold=threshold,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics or {},
        "best_val_metrics": best_val_metrics or {},
    }


def evaluate_test_split(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        threshold=threshold,
    )


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    optimizer_name = config.name.lower()
    if optimizer_name == "adam":
        optimizer_cls = torch.optim.Adam
    elif optimizer_name == "adamw":
        optimizer_cls = torch.optim.AdamW
    elif optimizer_name == "sgd":
        optimizer_cls = torch.optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer '{config.name}'.")

    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = 0.9

    return optimizer_cls(model.parameters(), **optimizer_kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_run_summary_text(
    *,
    best_epoch: int,
    best_val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> str:
    summary = (
        f"Best validation loss was {best_val_metrics['loss']:.4f} with "
        f"validation F1 {best_val_metrics['f1']:.4f} at epoch {best_epoch}."
    )
    if test_metrics is None:
        return f"{summary} Test evaluation was skipped for this run."
    return (
        f"{summary} Final test loss was {test_metrics['loss']:.4f} with "
        f"test F1 {test_metrics['f1']:.4f}."
    )


def run_training_pipeline(config: TrainingConfig) -> dict[str, Any]:
    artifacts = initialize_run_artifacts(
        artifact_root=config.artifact_root,
        run_name=config.run_name,
    )
    config = replace(config, run_name=artifacts.run_name)
    write_json(artifacts.config_path, config.to_dict())

    set_seed(config.seed)
    start_time = time.time()

    device = get_device()
    try:
        dataloaders, pos_weight, input_dim = build_dataloaders(config)
    except ConnectionError as exc:
        raise SystemExit(
            "Failed to download the Adult dataset. "
            "Check your internet connection or add a local dataset-loading fallback."
        ) from exc

    model = AdultIncomeMLP(
        input_dim=input_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        activation=config.model.activation,
        use_batch_norm=config.model.use_batch_norm,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = build_optimizer(model, config.optimizer)

    print(f"Using device: {device}")
    print(f"Run name: {config.run_name}")
    print(f"Artifacts: {artifacts.run_dir}")
    print(model)

    training_result = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
        threshold=config.threshold,
    )

    history_payload = {
        "run_name": config.run_name,
        "threshold": config.threshold,
        "epochs": training_result["history"],
    }
    write_json(artifacts.history_path, history_payload)

    test_metrics: dict[str, float] | None = None
    if config.final_evaluation:
        test_metrics = evaluate_test_split(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

    duration_seconds = round(time.time() - start_time, 2)
    best_val_metrics = training_result["best_val_metrics"]
    summary = {
        "run_name": config.run_name,
        "run_dir": str(artifacts.run_dir),
        "config": config.to_dict(),
        "device": str(device),
        "input_dim": input_dim,
        "parameter_count": count_parameters(model),
        "best_epoch": training_result["best_epoch"],
        "epochs_completed": len(training_result["history"]),
        "best_train_metrics": training_result["best_train_metrics"],
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "duration_seconds": duration_seconds,
        "summary_text": build_run_summary_text(
            best_epoch=training_result["best_epoch"],
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
        ),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "summary": summary,
        },
        artifacts.checkpoint_path,
    )
    leaderboard = update_leaderboard(artifacts, summary)
    current_rank = next(
        (
            entry["rank"]
            for entry in leaderboard
            if entry.get("run_name") == config.run_name
        ),
        None,
    )
    summary["leaderboard_rank"] = current_rank
    write_json(artifacts.summary_path, summary)

    print("\nRun summary")
    print(summary["summary_text"])
    print(f"Leaderboard rank: {current_rank if current_rank is not None else 'n/a'}")
    if test_metrics is not None:
        print("\nFinal test metrics")
        print(f"loss:      {test_metrics['loss']:.4f}")
        print(f"accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"precision: {test_metrics['precision']:.4f}")
        print(f"recall:    {test_metrics['recall']:.4f}")
        print(f"f1:        {test_metrics['f1']:.4f}")

    return summary


def parse_args() -> TrainingConfig:
    parser = ArgumentParser(description="Train the Adult Income MLP and store run artifacts.")
    parser.add_argument("--run-name", help="Optional run name. A timestamped one is generated by default.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for data splitting and training.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience based on validation loss.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold used for classification metrics.")
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Evaluate the test split and store final test metrics for this run.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Directory where run artifacts and the leaderboard will be written.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[256, 128, 64],
        help="Hidden layer widths. Pass no values after the flag for a linear model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout applied after each hidden layer.",
    )
    parser.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "leaky_relu", "gelu", "elu"],
        help="Hidden-layer activation.",
    )
    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable batch normalization in hidden layers.",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer used for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Optimizer weight decay.",
    )
    args = parser.parse_args()

    return TrainingConfig(
        run_name=args.run_name,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        threshold=args.threshold,
        final_evaluation=args.final_evaluation,
        artifact_root=args.artifact_root,
        model=ModelConfig(
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            activation=args.activation,
            use_batch_norm=not args.no_batch_norm,
        ),
        optimizer=OptimizerConfig(
            name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )


def main() -> None:
    config = parse_args()
    summary = run_training_pipeline(config)
    print(f"\nSaved summary to {Path(summary['run_dir']) / 'summary.json'}")


if __name__ == "__main__":
    main()
