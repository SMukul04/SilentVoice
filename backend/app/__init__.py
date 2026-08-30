"""SilentVoice FastAPI application package."""

from backend.app.main import app
from backend.app.schemas import PredictionRequest, PredictionResponse

__all__ = ["app", "PredictionRequest", "PredictionResponse"]
