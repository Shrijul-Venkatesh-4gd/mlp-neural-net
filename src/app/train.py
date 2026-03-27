from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import TrainingConfig
from app.data import load_preprocessed_adult_data
from app.model import AdultIncomeMLP


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
    prepared = load_preprocessed_adult_data(random_state=config.random_state)

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
    metrics = compute_binary_metrics(epoch_logits, epoch_targets)
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
) -> None:
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
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


def evaluate_test_split(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )


def run_training_pipeline(config: TrainingConfig) -> dict[str, float]:
    set_seed(config.random_state)

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
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Using device: {device}")
    print(model)

    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
    )

    test_metrics = evaluate_test_split(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test metrics")
    print(f"loss:      {test_metrics['loss']:.4f}")
    print(f"accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"precision: {test_metrics['precision']:.4f}")
    print(f"recall:    {test_metrics['recall']:.4f}")
    print(f"f1:        {test_metrics['f1']:.4f}")
    return test_metrics
