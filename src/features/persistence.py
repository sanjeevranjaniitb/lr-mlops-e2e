"""Persistence utilities for feature engineering artifacts."""

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def save_artifact(obj: object, path: str | Path) -> None:
    """Save any picklable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact to {path}")


def load_artifact(path: str | Path) -> object:
    """Load a previously saved artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    logger.info(f"Loaded artifact from {path}")
    return obj
