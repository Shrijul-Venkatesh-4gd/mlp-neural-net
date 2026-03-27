from __future__ import annotations

import pandas as pd

from utils.data.data_loader import load_preprocessed_adult_data


def main() -> None:
    data = load_preprocessed_adult_data()

    train_frame = pd.DataFrame(data.X_train, columns=data.feature_names)
    val_frame = pd.DataFrame(data.X_val, columns=data.feature_names)
    test_frame = pd.DataFrame(data.X_test, columns=data.feature_names)

    print("Preprocessed Adult dataset for MLP")
    print(f"Train shape: {train_frame.shape}, target shape: {data.y_train.shape}")
    print(f"Validation shape: {val_frame.shape}, target shape: {data.y_val.shape}")
    print(f"Test shape: {test_frame.shape}, target shape: {data.y_test.shape}")
    print(f"Class weights: {data.class_weights}")
    print()

    print("Train preview:")
    print(train_frame.head().to_string())
    print()

    print("Train targets preview:")
    print(pd.Series(data.y_train[:5], name="income").to_string(index=False))
    print()

    print("Feature names:")
    print(pd.Series(data.feature_names, name="feature").to_string(index=False))


if __name__ == "__main__":
    main()
