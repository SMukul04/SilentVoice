"""SilentVoice FastAPI application package."""

from backend.app.main import app
from backend.app.schemas import PredictionRequest, PredictionResponse, APIErrorResponse

__all__ = ["app", "PredictionRequest", "PredictionResponse", "APIErrorResponse"]
