# mlp-neural-net

PyTorch MLP baseline for the UCI Adult Income dataset:
https://archive.ics.uci.edu/dataset/2/adult

## Project layout

```text
src/app/
  cli.py
  config.py
  model.py
  training.py
  data/
    data_loader.py
    preprocessing.py
    eda/
      generate_report.py
main.py
docs/
```

## Run training

Use the thin root entrypoint:

```bash
.venv/bin/python main.py
```

Or run the package directly:

```bash
PYTHONPATH=src .venv/bin/python -m app.cli
```

If you are using `uv`, the script entrypoint is:

```bash
uv run adult-income-train
```

## Generate the EDA report

```bash
PYTHONPATH=src .venv/bin/python -m app.data.eda.generate_report
```

## Dataset access

The dataset loader uses `ucimlrepo` and fetches dataset id `2`.
