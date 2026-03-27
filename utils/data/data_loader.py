from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ucimlrepo import fetch_ucirepo

from utils.data.preprocessing import PreparedAdultData, prepare_adult_mlp_data

ADULT_DATASET_ID = 2


@dataclass(frozen=True)
class AdultDataset:
    metadata: dict
    variables: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series

    @property
    def frame(self) -> pd.DataFrame:
        dataset = self.features.copy()
        dataset[self.target.name or "income"] = self.target
        return dataset

    @property
    def cleaned_frame(self) -> pd.DataFrame:
        dataset = self.features.copy()
        dataset["income"] = normalize_income_labels(self.target)
        return dataset


def normalize_income_labels(target: pd.Series) -> pd.Series:
    normalized = target.replace(
        {
            "<=50K.": "<=50K",
            ">50K.": ">50K",
        }
    )
    return normalized.rename(target.name or "income")


def load_adult_dataset() -> AdultDataset:
    dataset = fetch_ucirepo(id=ADULT_DATASET_ID)
    target = dataset.data.targets.iloc[:, 0].copy()

    return AdultDataset(
        metadata=dataset.metadata,
        variables=dataset.variables.copy(),
        features=dataset.data.features.copy(),
        target=target,
    )


def load_preprocessed_adult_data(
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> PreparedAdultData:
    dataset = load_adult_dataset()
    normalized_target = normalize_income_labels(dataset.target)

    return prepare_adult_mlp_data(
        dataset.features,
        normalized_target,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
