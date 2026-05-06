"""Feature engineering and scaling pipeline."""

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature scaling and transformation."""

    def __init__(self, scaling: str = "standard"):
        self.scaling = scaling
        self.scaler = self._create_scaler()
        self._is_fitted = False

    def _create_scaler(self) -> StandardScaler | MinMaxScaler | None:
        if self.scaling == "standard":
            return StandardScaler()
        elif self.scaling == "minmax":
            return MinMaxScaler()
        elif self.scaling == "none":
            return None
        else:
            raise ValueError(f"Unknown scaling: {self.scaling}")

    def fit(self, X: np.ndarray) -> "FeatureEngineer":
        """Fit the scaler on training data."""
        if self.scaler is not None:
            self.scaler.fit(X)
            logger.info(f"Fitted {self.scaling} scaler on {X.shape[1]} features")
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using the fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Must be fitted before transform")
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
