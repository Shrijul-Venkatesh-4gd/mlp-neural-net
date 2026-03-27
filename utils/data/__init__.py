"""Utilities for loading and preprocessing project datasets."""

from utils.data.data_loader import (
    AdultDataset,
    load_adult_dataset,
    load_preprocessed_adult_data,
)
from utils.data.preprocessing import (
    PreparedAdultData,
    normalize_income_labels,
    prepare_adult_mlp_data,
)

__all__ = [
    "AdultDataset",
    "PreparedAdultData",
    "load_adult_dataset",
    "load_preprocessed_adult_data",
    "normalize_income_labels",
    "prepare_adult_mlp_data",
]
