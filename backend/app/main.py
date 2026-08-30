"""FastAPI application entry point for SilentVoice."""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse

from backend.app.schemas import PredictionRequest, PredictionResponse
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService


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

@app.get("/")
def read_root() -> dict[str, str]:
    """Return the API welcome message."""
    return {"message": "Welcome to SilentVoice API"}


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
    try:
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
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


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

