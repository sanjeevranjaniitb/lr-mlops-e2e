"""Data preprocessing and splitting utilities."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for train/test split data."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str


def split_data(
    df: pd.DataFrame,
    target_column: str = "MedHouseVal",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Split DataFrame into train and test sets."""
    feature_names = [col for col in df.columns if col != target_column]
    X = df[feature_names].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_name=target_column,
    )
