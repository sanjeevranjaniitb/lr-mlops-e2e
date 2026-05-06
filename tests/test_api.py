"""Tests for the FastAPI inference server."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.features.engineering import FeatureEngineer
from src.serving.app import app, model_state


@pytest.fixture
def mock_model():
    """Create a mock trained model for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = np.random.randn(100)

    model = LinearRegression()
    model.fit(X, y)

    scaler = FeatureEngineer(scaling="standard")
    scaler.fit(X)

    metadata = {
        "feature_names": [f"feature_{i}" for i in range(8)],
        "model_type": "linear_regression",
        "hyperparameters": {"fit_intercept": True},
        "scaling": "standard",
        "metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5, "mse": 1.0},
    }

    return {"model": model, "scaler": scaler, "metadata": metadata}


@pytest.fixture
def client(mock_model):
    """Create test client with loaded model."""
    model_state.update(mock_model)
    with TestClient(app) as c:
        yield c
    model_state.clear()


def test_health_check(client):
    """Test health endpoint with loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_check_no_model():
    """Test health endpoint without model."""
    from unittest.mock import patch

    with patch(
        "src.serving.app.load_model_artifacts",
        side_effect=FileNotFoundError("no model"),
    ):
        model_state.clear()
        with TestClient(app) as c:
            response = c.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False


def test_predict(client):
    """Test prediction endpoint."""
    payload = {"features": [[1.0] * 8]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], float)


def test_predict_batch(client):
    """Test batch prediction."""
    payload = {"features": [[1.0] * 8, [2.0] * 8, [3.0] * 8]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3


def test_predict_wrong_features(client):
    """Test prediction with wrong number of features."""
    payload = {"features": [[1.0, 2.0]]}  # Only 2 features, need 8
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_no_model():
    """Test prediction without loaded model."""
    from unittest.mock import patch

    with patch(
        "src.serving.app.load_model_artifacts",
        side_effect=FileNotFoundError("no model"),
    ):
        model_state.clear()
        with TestClient(app) as c:
            payload = {"features": [[1.0] * 8]}
            response = c.post("/predict", json=payload)
            assert response.status_code == 503


def test_model_info(client):
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data
    assert "model_type" in data
    assert "metrics" in data
