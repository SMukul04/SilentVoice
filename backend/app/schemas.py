"""Pydantic schemas for the SilentVoice backend API."""

import math
from typing import List

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Schema for a single landmark feature vector request."""
    
    features: List[float] = Field(
        ...,
        description="126-dimensional landmark feature vector",
        min_length=126,
        max_length=126
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        """Ensure all feature values are finite numeric values."""
        for val in v:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Feature values must be finite numbers (no NaN or Infinity)")
        return v


class PredictionResponse(BaseModel):
    """Schema for the backend prediction result."""
    
    predicted_index: int = Field(..., ge=0, description="Index of the predicted class")
    predicted_class: str = Field(..., min_length=1, description="Name of the predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction")
    probabilities: List[float] = Field(..., description="List of probabilities for all classes")
    sequence_ready: bool = Field(..., description="Whether the sequence buffer was ready for prediction")
    stable: bool = Field(..., description="Whether the prediction is considered stable")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: List[float]) -> List[float]:
        """Ensure all probability values are valid."""
        for val in v:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Probability values must be finite numbers")
            if val < 0.0 or val > 1.0:
                raise ValueError("Probability values must be between 0.0 and 1.0")
        return v
