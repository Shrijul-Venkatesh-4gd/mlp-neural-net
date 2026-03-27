from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


MISSING_LIKE_COLUMNS = ["workclass", "occupation", "native-country"]
CATEGORICAL_COLUMNS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUMERIC_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
DROP_COLUMNS = ["education", "fnlwgt"]


@dataclass(frozen=True)
class PreparedAdultData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]
    class_weights: dict[int, float]
    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame
    snapshot: pd.DataFrame


def _replace_missing_like_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in MISSING_LIKE_COLUMNS:
        cleaned[column] = cleaned[column].replace("?", np.nan)
    return cleaned


def _log1p_capital_features(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()
    for column in ["capital-gain", "capital-loss"]:
        transformed[column] = np.log1p(transformed[column])
    return transformed


def _preserve_feature_names(
    transformer: FunctionTransformer, input_features: list[str] | np.ndarray
) -> list[str] | np.ndarray:
    return input_features


def clean_adult_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = _replace_missing_like_values(frame)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.drop(columns=DROP_COLUMNS)
    return cleaned


def encode_income_target(target: pd.Series) -> pd.Series:
    encoded = target.map({"<=50K": 0, ">50K": 1})
    if encoded.isna().any():
        raise ValueError("Target contains unexpected labels after normalization.")
    return encoded.astype(int)


def build_mlp_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            (
                "log_capital",
                FunctionTransformer(
                    _log1p_capital_features,
                    validate=False,
                    feature_names_out=_preserve_feature_names,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def _split_validation_share(test_size: float, val_size: float) -> float:
    remaining_share = 1.0 - test_size
    if remaining_share <= 0:
        raise ValueError("test_size must be less than 1.0")
    return val_size / remaining_share


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    n_classes = classes.shape[0]
    return {
        int(class_label): float(total / (n_classes * count))
        for class_label, count in zip(classes, counts, strict=True)
    }


def build_preprocessed_snapshot(
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    sample_size: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    split_frames = []
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_frame = pd.DataFrame(X_split, columns=feature_names)
        split_frame["target"] = y_split.astype(int)
        split_frame["split"] = split_name
        split_frames.append(split_frame)

    combined = pd.concat(split_frames, ignore_index=True)
    n_rows = min(sample_size, len(combined))
    return combined.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


def prepare_adult_mlp_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> PreparedAdultData:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to less than 1.")

    dataset = features.copy()
    dataset["income"] = target
    dataset = clean_adult_dataframe(dataset)

    X = dataset.drop(columns=["income"])
    y = encode_income_target(dataset["income"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_share = _split_validation_share(test_size, val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_share,
        random_state=random_state,
        stratify=y_train_full,
    )

    preprocessor = build_mlp_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    class_weights = compute_class_weights(y_train.to_numpy())
    snapshot = build_preprocessed_snapshot(
        X_train=np.asarray(X_train_processed, dtype=np.float32),
        X_val=np.asarray(X_val_processed, dtype=np.float32),
        X_test=np.asarray(X_test_processed, dtype=np.float32),
        y_train=y_train.to_numpy(dtype=np.float32),
        y_val=y_val.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=np.float32),
        feature_names=feature_names,
        sample_size=50,
        random_state=random_state,
    )

    return PreparedAdultData(
        X_train=np.asarray(X_train_processed, dtype=np.float32),
        X_val=np.asarray(X_val_processed, dtype=np.float32),
        X_test=np.asarray(X_test_processed, dtype=np.float32),
        y_train=y_train.to_numpy(dtype=np.float32),
        y_val=y_val.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=np.float32),
        preprocessor=preprocessor,
        feature_names=feature_names,
        class_weights=class_weights,
        raw_train=X_train.reset_index(drop=True),
        raw_val=X_val.reset_index(drop=True),
        raw_test=X_test.reset_index(drop=True),
        snapshot=snapshot,
    )
