from __future__ import annotations

import pandas as pd

from utils.data.data_loader import load_preprocessed_adult_data


def main() -> None:
    data = load_preprocessed_adult_data()

    train_frame = pd.DataFrame(data.X_train, columns=data.feature_names)
    val_frame = pd.DataFrame(data.X_val, columns=data.feature_names)
    test_frame = pd.DataFrame(data.X_test, columns=data.feature_names)

    print(data.snapshot)

if __name__ == "__main__":
    main()
