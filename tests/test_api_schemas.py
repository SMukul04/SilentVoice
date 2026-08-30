"""Unit tests for the SilentVoice API schemas."""

import math
import json
import pytest
from pydantic import ValidationError

from backend.app.schemas import PredictionRequest, PredictionResponse

def test_valid_prediction_request() -> None:
    """Test 1: Valid PredictionRequest."""
    features = [0.1] * 126
    req = PredictionRequest(features=features)
    assert len(req.features) == 126
    assert req.features[0] == 0.1

def test_invalid_feature_count() -> None:
    """Test 2: Invalid feature count."""
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.1] * 125)
        
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.1] * 127)

def test_invalid_feature_dimensions() -> None:
    """Test 3: Invalid feature dimensions."""
    # Nested list input
    with pytest.raises(ValidationError):
        PredictionRequest(features=[[0.1]] * 126) # type: ignore

def test_invalid_numeric_values() -> None:
    """Test 4: Invalid numeric values."""
    features_nan = [0.1] * 125 + [float('nan')]
    with pytest.raises(ValidationError):
        PredictionRequest(features=features_nan)
        
    features_inf = [0.1] * 125 + [float('inf')]
    with pytest.raises(ValidationError):
        PredictionRequest(features=features_inf)
        
    features_ninf = [0.1] * 125 + [float('-inf')]
    with pytest.raises(ValidationError):
        PredictionRequest(features=features_ninf)

def test_invalid_feature_types() -> None:
    """Test 5: Invalid feature types."""
    features_str = [0.1] * 125 + ["string"]
    with pytest.raises(ValidationError):
        PredictionRequest(features=features_str) # type: ignore
        
    features_none = [0.1] * 125 + [None]
    with pytest.raises(ValidationError):
        PredictionRequest(features=features_none) # type: ignore

def test_valid_prediction_response() -> None:
    """Test 6: Valid PredictionResponse."""
    resp = PredictionResponse(
        predicted_index=1,
        predicted_class="hello",
        confidence=0.99,
        probabilities=[0.01, 0.99],
        sequence_ready=True,
        stable=True
    )
    assert resp.predicted_index == 1
    assert resp.predicted_class == "hello"

def test_invalid_confidence() -> None:
    """Test 7: Invalid confidence."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=-0.1,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )
        
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=1.1,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )

def test_invalid_probabilities() -> None:
    """Test 8: Invalid probabilities."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=0.99,
            probabilities=[-0.1, 0.99],
            sequence_ready=True,
            stable=True
        )
        
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=0.99,
            probabilities=[0.01, 1.1],
            sequence_ready=True,
            stable=True
        )

def test_invalid_predicted_class() -> None:
    """Test 9: Invalid predicted class."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="",
            confidence=0.99,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )

def test_json_serialization() -> None:
    """Test 10: JSON serialization."""
    resp = PredictionResponse(
        predicted_index=1,
        predicted_class="hello",
        confidence=0.99,
        probabilities=[0.01, 0.99],
        sequence_ready=True,
        stable=True
    )
    
    json_str = resp.model_dump_json()
    reconstructed = PredictionResponse.model_validate_json(json_str)
    
    assert reconstructed.predicted_index == resp.predicted_index
    assert reconstructed.predicted_class == resp.predicted_class
    assert reconstructed.confidence == resp.confidence
    assert reconstructed.probabilities == resp.probabilities
    assert reconstructed.sequence_ready == resp.sequence_ready
    assert reconstructed.stable == resp.stable

def test_schema_compatibility() -> None:
    """Test 11: Schema compatibility with PredictionService output."""
    service_output = {
        "predicted_index": 0,
        "predicted_class": "alive",
        "confidence": 0.94,
        "probabilities": [0.94, 0.06],
        "sequence_ready": True,
        "stable": True
    }
    
    resp = PredictionResponse(**service_output)
    assert resp.predicted_index == 0
    assert resp.predicted_class == "alive"
    assert resp.confidence == 0.94

def test_invalid_response_structure() -> None:
    """Test 12: Invalid response structure."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            # missing confidence
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        ) # type: ignore
