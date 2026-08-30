"""Service for orchestrating stateful sign language predictions."""

from __future__ import annotations

import logging
from typing import Any

from backend.services.model_service import ModelService
from backend.realtime.prediction_stabilizer import PredictionStabilizer


logger = logging.getLogger(__name__)


class PredictionService:
    """Coordinates real-time inference between the FastAPI backend and model logic.
    
    This service maintains the lifecycle of inference components and provides
    a stateful interface for adding landmarks, performing smoothed predictions,
    and handling prediction sequence resets.
    """

    def __init__(
        self,
        model_service: ModelService,
        stabilizer: PredictionStabilizer | None = None
    ) -> None:
        """Initialize the PredictionService.
        
        Args:
            model_service: The ModelService instance containing the InferenceEngine.
            stabilizer: Optional PredictionStabilizer for smoothing predictions.
                If not provided, a default instance is created.
        """
        if not isinstance(model_service, ModelService):
            raise TypeError("model_service must be a ModelService instance")
        
        self.model_service = model_service
        self.stabilizer = stabilizer if stabilizer is not None else PredictionStabilizer()
        logger.info("PredictionService initialized")

    def add_landmarks(self, features: Any) -> dict[str, Any]:
        """Add a landmark vector to the sequence buffer.
        
        Args:
            features: A 126-d landmark vector (array-like).
            
        Returns:
            dict: Sequence readiness information (sequence_ready, sequence_length).
            
        Raises:
            TypeError, ValueError: If features are invalid or incorrectly shaped.
            RuntimeError: If model/metadata loading fails.
        """
        engine = self.model_service.get_inference_engine()
        engine.add_landmarks(features)
        
        return {
            "sequence_ready": engine.is_ready(),
            "sequence_length": engine.sequence_count
        }

    def is_ready(self) -> bool:
        """Check if the inference engine is ready for prediction.
        
        Returns:
            bool: True if the sequence buffer is full.
            
        Raises:
            RuntimeError: If model/metadata loading fails.
        """
        engine = self.model_service.get_inference_engine()
        return engine.is_ready()

    def predict(self) -> dict[str, Any]:
        """Run inference on the current sequence and stabilize the output.
        
        Returns:
            dict: The stabilized prediction including index, class, confidence,
                  probabilities, sequence_ready, and stable flags.
                  
        Raises:
            RuntimeError: If the sequence is not ready or model is not loaded.
            ValueError: If the prediction output is invalid.
        """
        engine = self.model_service.get_inference_engine()
        if not engine.is_ready():
            raise RuntimeError("Sequence is not ready for prediction")

        raw_prediction = engine.predict()
        stabilized = self.stabilizer.add_prediction(raw_prediction)
        
        return {
            "predicted_index": stabilized["predicted_index"],
            "predicted_class": stabilized["predicted_class"],
            "confidence": stabilized["confidence"],
            "probabilities": stabilized["probabilities"],
            "sequence_ready": True,
            "stable": self.stabilizer.is_stable()
        }

    def reset(self) -> None:
        """Reset the inference sequence buffer and stabilizer history."""
        try:
            engine = self.model_service.get_inference_engine()
            engine.reset()
        except Exception as e:
            logger.warning("Could not reset inference engine: %s", e)
            
        self.stabilizer.reset()
        logger.debug("PredictionService reset complete")
