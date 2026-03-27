from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ucimlrepo import fetch_ucirepo

from app.data.preprocessing import (
    PreparedAdultData,
    clean_adult_dataframe,
    normalize_income_labels,
    prepare_adult_mlp_data,
)

ADULT_DATASET_ID = 2


@dataclass(frozen=True)
class AdultDataset:
    metadata: dict
    variables: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series

    @classmethod
    def load(cls, dataset_id: int = ADULT_DATASET_ID) -> "AdultDataset":
        dataset = fetch_ucirepo(id=dataset_id)
        target = dataset.data.targets.iloc[:, 0].copy()

        return cls(
            metadata=dataset.metadata,
            variables=dataset.variables.copy(),
            features=dataset.data.features.copy(),
            target=target,
        )

    @property
    def target_name(self) -> str:
        return self.target.name or "income"

    def get_variables(self) -> pd.DataFrame:
        return self.variables.copy()

    def get_features(self) -> pd.DataFrame:
        return self.features.copy()

    def get_target(self, *, normalized: bool = False) -> pd.Series:
        target = self.target.copy()
        if normalized:
            return normalize_income_labels(target)
        return target.rename(self.target_name)

    @property
    def frame(self) -> pd.DataFrame:
        return self.to_frame()

    @property
    def cleaned_frame(self) -> pd.DataFrame:
        return self.get_cleaned_frame()

    def to_frame(self, *, normalized_target: bool = False) -> pd.DataFrame:
        dataset = self.get_features()
        dataset[self.target_name] = self.get_target(normalized=normalized_target)
        return dataset

    def get_cleaned_frame(self) -> pd.DataFrame:
        return clean_adult_dataframe(self.to_frame(normalized_target=True))

    def preprocess_for_mlp(
        self,
        *,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> PreparedAdultData:
        return prepare_adult_mlp_data(
            self.get_features(),
            self.get_target(normalized=True),
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )


def load_adult_dataset() -> AdultDataset:
    return AdultDataset.load()


def load_preprocessed_adult_data(
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> PreparedAdultData:
    return AdultDataset.load().preprocess_for_mlp(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

