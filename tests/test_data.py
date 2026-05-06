"""Tests for data loading and preprocessing."""

import pandas as pd

from src.data.loader import load_california_housing
from src.data.preprocessor import SplitData, split_data


def test_load_california_housing():
    """Test that California Housing dataset loads correctly."""
    df = load_california_housing()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "MedHouseVal" in df.columns


def test_split_data():
    """Test train/test splitting."""
    df = load_california_housing()
    data = split_data(df, test_size=0.2, random_state=42)

    assert isinstance(data, SplitData)
    assert data.X_train.shape[0] > data.X_test.shape[0]
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert len(data.y_train) == data.X_train.shape[0]
    assert len(data.y_test) == data.X_test.shape[0]
    assert data.target_name == "MedHouseVal"


def test_split_data_ratio():
    """Test that split ratio is approximately correct."""
    df = load_california_housing()
    data = split_data(df, test_size=0.3, random_state=42)

    total = data.X_train.shape[0] + data.X_test.shape[0]
    test_ratio = data.X_test.shape[0] / total
    assert abs(test_ratio - 0.3) < 0.01
