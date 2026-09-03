"""FastAPI application entry point for SilentVoice."""

import logging
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.schemas import PredictionRequest, PredictionResponse, APIErrorResponse
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService


logger = logging.getLogger(__name__)

app = FastAPI(
    title="SilentVoice API",
    description="Backend API for real-time sign language recognition.",
    version="1.0.0",
)

# Global service instance to persist the sequence buffer across HTTP requests
_model_service = ModelService()
_prediction_service = PredictionService(model_service=_model_service)

def get_prediction_service() -> PredictionService:
    return _prediction_service


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI/Pydantic validation errors (HTTP 422)."""
    return JSONResponse(
        status_code=422,
        content=APIErrorResponse(
            success=False,
            error="Validation error",
            detail="Invalid prediction request"
        ).model_dump()
    )


@app.exception_handler(FileNotFoundError)
async def model_missing_exception_handler(request: Request, exc: FileNotFoundError):
    """Handle model loading failures (HTTP 503)."""
    logger.exception("Model unavailable")
    return JSONResponse(
        status_code=503,
        content=APIErrorResponse(
            success=False,
            error="Model unavailable",
            detail="The recognition model is currently unavailable."
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other unexpected errors (HTTP 500)."""
    logger.exception("Unexpected backend failure")
    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            success=False,
            error="Prediction failed",
            detail="Unable to process the prediction."
        ).model_dump()
    )


from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Configure static files and templates
frontend_dir = Path("frontend")
app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")
app.mount("/models", StaticFiles(directory=Path("models")), name="models")
templates = Jinja2Templates(directory=frontend_dir / "templates")


@app.get("/")
def read_root() -> dict[str, str]:
    """Return the API welcome message."""
    return {"message": "Welcome to SilentVoice API"}


@app.get("/app")
def serve_frontend(request: Request):
    """Serve the SilentVoice frontend application."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """Accept one landmark vector and return a prediction if the sequence is ready."""
    status = service.add_landmarks(request.features)
    
    if not status.get("sequence_ready", False):
        # Sequence not ready, return a placeholder Response
        return PredictionResponse(
            predicted_index=0,
            predicted_class="unknown",
            confidence=0.0,
            probabilities=[],
            sequence_ready=False,
            stable=False
        )
        
    prediction = service.predict()
    return PredictionResponse(**prediction)


@app.post("/predict/reset")
def reset_prediction(
    service: PredictionService = Depends(get_prediction_service)
) -> dict:
    """Reset the prediction sequence and stabilizer history."""
    service.reset()
    return {
        "success": True,
        "message": "Prediction state reset"
    }
