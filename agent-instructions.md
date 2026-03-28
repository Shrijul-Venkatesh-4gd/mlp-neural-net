# Agent Instructions For Hyperparameter Tuning

## Purpose

You are an autonomous coding agent helping tune this project's training hyperparameters for the Adult Income MLP.

Your job is to improve validation performance by launching controlled training runs, reading the saved artifacts, and choosing the next set of training hyperparameters to try.

## Primary Objective

- Optimize for `best_val_f1` from `artifacts/leaderboard.json`.
- Use `best_val_loss` only as a secondary tie-breaker.
- Do not choose a winner based on test metrics.

## Correct Entrypoint

Use the training module CLI, not `main.py`.

Preferred command pattern:

```bash
uv run python -m app.train --run-name <name> [flags]
```

Why:

- `main.py` only runs the default config and does not expose the tuning controls you need.
- `app.train` exposes the tunable training flags.

## Allowed Search Space

Tune only these training hyperparameters:

- `--optimizer`: `adam`, `adamw`, `sgd`
- `--learning-rate`: start in the range `1e-4` to `3e-3`
- `--weight-decay`: start in the range `0.0` to `1e-3`
- `--batch-size`: try `128`, `256`, or `512`
- `--epochs`: typically `15` to `40`
- `--patience`: typically `3` to `8`
- `--threshold`: typically `0.35` to `0.70`

`seed` is not a score-optimization knob.

- Keep `--seed 42` during the first pass so runs are comparable.
- Use additional seeds only to verify the best few configs after they look promising.

## Do Not Tune These

Do not change model architecture or preprocessing.

Forbidden flags and settings:

- `--hidden-dims`
- `--dropout`
- `--activation`
- `--no-batch-norm`
- dataset loading logic
- preprocessing logic
- feature engineering
- model code

Do not edit source code while tuning unless the user explicitly asks for a code change.

## Run Protocol

1. Read `artifacts/leaderboard.json` before proposing the next runs.
2. Propose a small batch of runs, usually `2` to `4`.
3. Change only `1` or `2` hyperparameters at a time when possible.
4. Give each run a readable `--run-name`.
5. During tuning, do not use `--final-evaluation`.
6. Compare runs using validation metrics only.
7. After finding a strong config, rerun it across multiple seeds for robustness.
8. Only after locking the final config should you run one final evaluation with `--final-evaluation`.

## Leaderboard Interpretation

Treat the leaderboard as the source of truth for ranking experiments.

- Higher `best_val_f1` is better.
- Lower `best_val_loss` is better only when `best_val_f1` is tied or nearly tied.
- Ignore `test_f1` and `test_loss` during tuning.
- If two rows are duplicates with the same config and seed, do not treat that as evidence of improvement.

## Prompt Guardrails

You must follow these guardrails:

- Do not use the test set to choose hyperparameters.
- Do not modify model architecture parameters.
- Do not change code just to chase a metric.
- Do not delete old run artifacts or rewrite the leaderboard manually.
- Do not rerun the exact same config with the same seed unless a previous run failed or reproducibility is being checked.
- Do not claim a config is better without citing the relevant leaderboard entry or saved summary.
- Do not launch unbounded sweeps. Keep the search deliberate and explain why each run exists.

## Suggested Search Order

Use this order unless the leaderboard strongly suggests otherwise:

1. Tune `threshold`
2. Tune `learning-rate`
3. Tune `optimizer`
4. Tune `weight-decay`
5. Tune `batch-size`
6. Tune `patience` and `epochs`

## Example Commands

```bash
uv run python -m app.train --run-name tune_thr055 --threshold 0.55
uv run python -m app.train --run-name tune_lr3e4 --learning-rate 3e-4
uv run python -m app.train --run-name tune_adamw --optimizer adamw --learning-rate 3e-4 --weight-decay 1e-4
uv run python -m app.train --run-name tune_bs128 --batch-size 128
```

## Finalization Rule

When a candidate looks best on validation:

1. Re-run it with multiple seeds.
2. Confirm it is consistently strong on validation.
3. Run exactly one final locked-in test evaluation with `--final-evaluation`.
4. Report the validation winner and the final held-out test result separately.
