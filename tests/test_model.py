"""Tests for model training and evaluation."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.models.train import create_model, evaluate_model, load_config


def test_load_config():
    """Test configuration loading."""
    config = load_config()
    assert "model" in config
    assert "data" in config
    assert "features" in config
    assert "training" in config


def test_create_linear_regression():
    """Test linear regression model creation."""
    config = {
        "model": {
            "type": "linear_regression",
            "hyperparameters": {"fit_intercept": True},
        }
    }
    model = create_model(config)
    assert isinstance(model, LinearRegression)


def test_create_invalid_model():
    """Test that invalid model type raises ValueError."""
    config = {
        "model": {
            "type": "invalid_model",
            "hyperparameters": {},
        }
    }
    with pytest.raises(ValueError):
        create_model(config)


def test_evaluate_model():
    """Test model evaluation returns expected metrics."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4]) + np.random.randn(100) * 0.1

    model = LinearRegression()
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mse" in metrics
    assert metrics["r2"] > 0.9  # Should fit well
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
