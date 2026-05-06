"""Data loading utilities for the ML pipeline."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

logger = logging.getLogger(__name__)


def load_california_housing() -> pd.DataFrame:
    """Load the California Housing dataset as a pandas DataFrame."""
    logger.info("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved data to {path}")
