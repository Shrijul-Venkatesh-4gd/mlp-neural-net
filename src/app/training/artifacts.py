from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LEADERBOARD_COLUMNS = [
    "rank",
    "run_name",
    "best_val_f1",
    "best_val_loss",
    "best_epoch",
    "epochs_completed",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "hidden_dims",
    "activation",
    "dropout",
    "threshold",
    "seed",
    "final_evaluation",
    "test_f1",
    "test_loss",
    "run_dir",
]


@dataclass(frozen=True)
class RunArtifactPaths:
    artifact_root: Path
    run_name: str
    run_dir: Path
    config_path: Path
    history_path: Path
    summary_path: Path
    checkpoint_path: Path
    leaderboard_json_path: Path
    leaderboard_csv_path: Path


def initialize_run_artifacts(
    artifact_root: str | Path,
    run_name: str | None = None,
) -> RunArtifactPaths:
    root = Path(artifact_root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resolved_name = base_name
    suffix = 2
    while (runs_dir / resolved_name).exists():
        resolved_name = f"{base_name}_{suffix:02d}"
        suffix += 1

    run_dir = runs_dir / resolved_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return RunArtifactPaths(
        artifact_root=root,
        run_name=resolved_name,
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        history_path=run_dir / "history.json",
        summary_path=run_dir / "summary.json",
        checkpoint_path=run_dir / "model.pt",
        leaderboard_json_path=root / "leaderboard.json",
        leaderboard_csv_path=root / "leaderboard.csv",
    )


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def update_leaderboard(artifacts: RunArtifactPaths, summary: dict[str, Any]) -> list[dict[str, Any]]:
    existing_entries: list[dict[str, Any]] = []
    if artifacts.leaderboard_json_path.exists():
        existing_entries = json.loads(artifacts.leaderboard_json_path.read_text(encoding="utf-8"))

    new_entry = build_leaderboard_entry(summary)
    deduped_entries = [
        entry for entry in existing_entries if entry.get("run_name") != new_entry["run_name"]
    ]
    deduped_entries.append(new_entry)
    deduped_entries.sort(
        key=lambda entry: (
            -float(entry["best_val_f1"]),
            float(entry["best_val_loss"]),
            entry["run_name"],
        )
    )

    ranked_entries: list[dict[str, Any]] = []
    for rank, entry in enumerate(deduped_entries, start=1):
        ranked_entry = dict(entry)
        ranked_entry["rank"] = rank
        ranked_entries.append(ranked_entry)

    write_json(artifacts.leaderboard_json_path, ranked_entries)
    write_leaderboard_csv(artifacts.leaderboard_csv_path, ranked_entries)
    return ranked_entries


def build_leaderboard_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary["config"]
    optimizer = config["optimizer"]
    model = config["model"]
    test_metrics = summary.get("test_metrics") or {}

    return {
        "rank": 0,
        "run_name": summary["run_name"],
        "best_val_f1": summary["best_val_metrics"]["f1"],
        "best_val_loss": summary["best_val_metrics"]["loss"],
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "optimizer": optimizer["name"],
        "learning_rate": optimizer["learning_rate"],
        "weight_decay": optimizer["weight_decay"],
        "hidden_dims": "-".join(str(dim) for dim in model["hidden_dims"]) or "linear",
        "activation": model["activation"],
        "dropout": model["dropout"],
        "threshold": config["threshold"],
        "seed": config["seed"],
        "final_evaluation": config["final_evaluation"],
        "test_f1": test_metrics.get("f1"),
        "test_loss": test_metrics.get("loss"),
        "run_dir": str(summary["run_dir"]),
    }


def write_leaderboard_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({column: entry.get(column) for column in LEADERBOARD_COLUMNS})
