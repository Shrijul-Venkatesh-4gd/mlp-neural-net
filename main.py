from __future__ import annotations

from utils.data.data_loader import AdultDataset


def main() -> None:
    dataset = AdultDataset.load()
    data = dataset.preprocess_for_mlp()

    print(data.snapshot)


if __name__ == "__main__":
    main()
