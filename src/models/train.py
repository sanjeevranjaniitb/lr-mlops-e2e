"""Model training pipeline with MLflow tracking."""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_california_housing
from src.data.preprocessor import split_data
from src.features.engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load project configuration."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict):
    """Create model instance from config."""
    model_type = config["model"]["type"]
    params = config["model"]["hyperparameters"]

    if model_type == "linear_regression":
        return LinearRegression(**params)
    elif model_type == "ridge":
        return Ridge(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
    }
    return metrics


def save_model_artifacts(
    model,
    feature_engineer: FeatureEngineer,
    feature_names: list[str],
    metrics: dict,
    config: dict,
) -> Path:
    """Save model and associated artifacts."""
    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature engineer (scaler)
    scaler_path = models_dir / "scaler.joblib"
    joblib.dump(feature_engineer, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "model_type": config["model"]["type"],
        "hyperparameters": config["model"]["hyperparameters"],
        "scaling": config["features"]["scaling"],
        "metrics": metrics,
    }
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    return models_dir


def _is_mlflow_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Check if MLflow tracking server is reachable."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5000
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, TimeoutError):
        return False


def train() -> dict:
    """Execute the full training pipeline."""
    config = load_config()
    training_config = config["training"]

    # Setup MLflow - probe server first, fall back to local file store
    remote_uri = training_config["mlflow_tracking_uri"]
    if _is_mlflow_server_reachable(remote_uri):
        logger.info(f"Using MLflow server at {remote_uri}")
        mlflow.set_tracking_uri(remote_uri)
    else:
        logger.info("MLflow server unavailable, using local file tracking")
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(training_config["experiment_name"])

    # Load data
    df = load_california_housing()

    # Split data
    data = split_data(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Feature engineering
    feature_engineer = FeatureEngineer(scaling=config["features"]["scaling"])
    X_train_scaled = feature_engineer.fit_transform(data.X_train)
    X_test_scaled = feature_engineer.transform(data.X_test)

    # Train model
    with mlflow.start_run():
        model = create_model(config)
        logger.info(f"Training {config['model']['type']}...")
        model.fit(X_train_scaled, data.y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test_scaled, data.y_test)
        logger.info(f"Metrics: {metrics}")

        # Log to MLflow
        mlflow.log_params(config["model"]["hyperparameters"])
        mlflow.log_param("scaling", config["features"]["scaling"])
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_metrics(metrics)

        # Save artifacts
        save_model_artifacts(
            model, feature_engineer, data.feature_names, metrics, config
        )

        logger.info("Training complete!")
        return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"\nFinal Metrics:\n{json.dumps(metrics, indent=2)}")
