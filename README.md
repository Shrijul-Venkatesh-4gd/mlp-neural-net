# mlp-neural-net

PyTorch MLP baseline for the UCI Adult Income dataset:
https://archive.ics.uci.edu/dataset/2/adult

## Project layout

```text
src/app/
  config.py
  model.py
  train.py
  data/
    data_loader.py
    preprocessing.py
    eda/
      generate_report.py
main.py
docs/
```

## Run training

```bash
.venv/bin/python main.py
```

## Generate the EDA report

```bash
PYTHONPATH=src .venv/bin/python -m app.data.eda.generate_report
```

## Dataset access

The dataset loader uses `ucimlrepo` and fetches dataset id `2`.
