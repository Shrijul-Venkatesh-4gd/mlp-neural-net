# mlp-neural-net

PyTorch MLP baseline for the UCI Adult Income dataset, with preprocessing, EDA reporting, and experiment-tracked training runs.

Dataset source: https://archive.ics.uci.edu/dataset/2/adult

## Overview

This repo trains a multilayer perceptron on the Adult Income classification task and stores reproducible run artifacts for comparison.

It includes:
- dataset loading through `ucimlrepo`
- preprocessing for tabular MLP training
- markdown EDA report generation
- CLI-based training with configurable hyperparameters
- per-run artifacts, checkpoints, and leaderboard files

## Project Layout

```text
src/app/
  config.py
  model.py
  train.py
  training/
    artifacts.py
  data/
    data_loader.py
    preprocessing.py
    eda/
      generate_report.py
artifacts/
  runs/
  leaderboard.json
  leaderboard.csv
docs/
  data/
main.py
```

## Requirements

- Python 3.10+
- `uv` recommended for running commands

Dependencies are defined in `pyproject.toml`.

## Install

```bash
uv sync
```

## Run Training

Preferred training entrypoint:

```bash
uv run python -m app.train --run-name baseline
```

Example tuned run:

```bash
uv run python -m app.train \
  --run-name tune_adamw_bs128_wd5e4_ep35_pat6_thr052 \
  --optimizer adamw \
  --learning-rate 5e-4 \
  --weight-decay 5e-4 \
  --batch-size 128 \
  --epochs 35 \
  --patience 6 \
  --threshold 0.52
```

Run a final locked evaluation only after selecting the winning config:

```bash
uv run python -m app.train --run-name final_eval --final-evaluation
```

## Generate The EDA Report

```bash
uv run python -m app.data.eda.generate_report
```

To write the report to a custom path:

```bash
uv run python -m app.data.eda.generate_report --output docs/data/eda_report.md
```

## Training Artifacts

Each training run creates a directory under `artifacts/runs/<run_name>/` with:
- `config.json`
- `history.json`
- `summary.json`
- `model.pt`

The repo also maintains:
- `artifacts/leaderboard.json`
- `artifacts/leaderboard.csv`

Runs are ranked primarily by validation F1 and secondarily by validation loss.

## Current Tuning Snapshot

Current strong config from the saved leaderboard:
- optimizer: `adamw`
- learning rate: `5e-4`
- weight decay: `5e-4`
- batch size: `128`
- epochs: `35`
- patience: `6`
- threshold: `0.52`

This tuning pass improved validation F1 over the baseline configuration.

## CLI Entrypoints

Installed scripts from `pyproject.toml`:

```bash
uv run adult-income-train --run-name baseline
uv run adult-income-eda
```

## Dataset Access

The dataset loader uses `ucimlrepo` and fetches UCI Adult dataset id `2`.
