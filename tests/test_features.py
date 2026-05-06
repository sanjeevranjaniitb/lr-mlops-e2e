"""Tests for feature engineering."""

import numpy as np
import pytest

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 4)


def test_standard_scaling(sample_data):
    """Test standard scaling produces zero mean and unit variance."""
    fe = FeatureEngineer(scaling="standard")
    transformed = fe.fit_transform(sample_data)

    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(transformed.std(axis=0), 1, atol=0.1)


def test_minmax_scaling(sample_data):
    """Test minmax scaling produces values in [0, 1]."""
    fe = FeatureEngineer(scaling="minmax")
    transformed = fe.fit_transform(sample_data)

    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0


def test_no_scaling(sample_data):
    """Test that 'none' scaling returns data unchanged."""
    fe = FeatureEngineer(scaling="none")
    transformed = fe.fit_transform(sample_data)

    assert np.array_equal(transformed, sample_data)


def test_transform_before_fit_raises():
    """Test that transforming before fitting raises an error."""
    fe = FeatureEngineer(scaling="standard")
    with pytest.raises(RuntimeError):
        fe.transform(np.array([[1, 2, 3]]))


def test_invalid_scaling():
    """Test that invalid scaling method raises ValueError."""
    with pytest.raises(ValueError):
        FeatureEngineer(scaling="invalid")
