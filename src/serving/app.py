"""FastAPI inference server for the trained model."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model state
model_state: dict = {}

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_model_artifacts() -> dict:
    """Load model, scaler, and metadata from disk."""
    model_path = MODELS_DIR / "model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    metadata_path = MODELS_DIR / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info("Model artifacts loaded successfully")
    return {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_state
    try:
        model_state = load_model_artifacts()
        logger.info("Model loaded and ready for inference")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please train the model first: python -m src.models.train")
    yield
    model_state.clear()


app = FastAPI(
    title="Linear Regression MLOps API",
    description="Production inference API for California Housing price prediction",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    features: list[list[float]] = Field(
        ...,
        description="2D array of features. Each inner list is one sample.",
        examples=[[[8.3, 41.0, 6.9, 1.02, 322.0, 2.55, 37.88, -122.23]]],
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    predictions: list[float]
    model_type: str
    feature_names: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_type: str | None = None
    metrics: dict | None = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    if not model_state:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=model_state["metadata"]["model_type"],
        metrics=model_state["metadata"]["metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate predictions for input features."""
    if not model_state:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    try:
        X = np.array(request.features)
        expected_features = len(model_state["metadata"]["feature_names"])

        if X.shape[1] != expected_features:
            raise HTTPException(
                status_code=422,
                detail=(f"Expected {expected_features} features, got {X.shape[1]}"),
            )

        # Apply scaling
        scaler: FeatureEngineer = model_state["scaler"]
        X_scaled = scaler.transform(X)

        # Predict
        predictions = model_state["model"].predict(X_scaled).tolist()

        return PredictionResponse(
            predictions=predictions,
            model_type=model_state["metadata"]["model_type"],
            feature_names=model_state["metadata"]["feature_names"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_state["metadata"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
