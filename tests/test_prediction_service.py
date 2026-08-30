"""Unit tests for the PredictionService."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from backend.realtime.prediction_stabilizer import PredictionStabilizer


@pytest.fixture
def mock_engine() -> MagicMock:
    """Provides a mocked InferenceEngine."""
    engine = MagicMock()
    engine.is_ready.return_value = False
    engine.sequence_count = 0
    return engine


@pytest.fixture
def mock_model_service(mock_engine: MagicMock) -> MagicMock:
    """Provides a mocked ModelService that returns the mock_engine."""
    service = MagicMock(spec=ModelService)
    service.get_inference_engine.return_value = mock_engine
    return service


def test_component_initialization(mock_model_service: MagicMock) -> None:
    """Test 1: Component initialization."""
    service = PredictionService(model_service=mock_model_service)
    assert service.model_service is mock_model_service
    assert isinstance(service.stabilizer, PredictionStabilizer)


def test_dependency_injection(mock_model_service: MagicMock) -> None:
    """Test 2: Dependency injection."""
    custom_stabilizer = PredictionStabilizer(window_size=3)
    service = PredictionService(model_service=mock_model_service, stabilizer=custom_stabilizer)
    assert service.stabilizer is custom_stabilizer


def test_landmark_forwarding(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 3: Landmark forwarding."""
    service = PredictionService(model_service=mock_model_service)
    
    mock_engine.sequence_count = 1
    mock_engine.is_ready.return_value = False
    
    features = np.random.rand(126).astype(np.float32)
    result = service.add_landmarks(features)
    
    mock_engine.add_landmarks.assert_called_once_with(features)
    assert result == {"sequence_ready": False, "sequence_length": 1}


def test_readiness_behavior(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 4: Readiness behavior."""
    service = PredictionService(model_service=mock_model_service)
    
    mock_engine.is_ready.return_value = False
    assert not service.is_ready()
    
    mock_engine.is_ready.return_value = True
    assert service.is_ready()


def test_prediction_before_ready(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 5: Prediction before ready."""
    service = PredictionService(model_service=mock_model_service)
    mock_engine.is_ready.return_value = False
    
    with pytest.raises(RuntimeError, match="Sequence is not ready for prediction"):
        service.predict()


def test_prediction_forwarding(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 6: Prediction forwarding."""
    service = PredictionService(model_service=mock_model_service)
    mock_engine.is_ready.return_value = True
    
    raw_prediction = {
        "predicted_index": 0,
        "predicted_class": "alive",
        "confidence": 0.95,
        "probabilities": [0.95, 0.05]
    }
    mock_engine.predict.return_value = raw_prediction
    
    result = service.predict()
    
    mock_engine.predict.assert_called_once()
    assert result["predicted_index"] == 0
    assert result["predicted_class"] == "alive"


def test_stabilized_prediction_output(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 7: Stabilized prediction output."""
    service = PredictionService(model_service=mock_model_service)
    mock_engine.is_ready.return_value = True
    
    raw_prediction = {
        "predicted_index": 1,
        "predicted_class": "hello",
        "confidence": 0.99,
        "probabilities": [0.01, 0.99]
    }
    mock_engine.predict.return_value = raw_prediction
    
    result = service.predict()
    
    expected_keys = {
        "predicted_index",
        "predicted_class",
        "confidence",
        "probabilities",
        "sequence_ready",
        "stable"
    }
    assert set(result.keys()) == expected_keys
    assert result["predicted_index"] == 1
    assert result["sequence_ready"] is True
    assert isinstance(result["stable"], bool)


def test_reset_behavior(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 8: Reset behavior."""
    custom_stabilizer = MagicMock(spec=PredictionStabilizer)
    service = PredictionService(model_service=mock_model_service, stabilizer=custom_stabilizer)
    
    service.reset()
    
    mock_engine.reset.assert_called_once()
    custom_stabilizer.reset.assert_called_once()


def test_invalid_landmark_handling(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 9: Invalid landmark handling."""
    service = PredictionService(model_service=mock_model_service)
    
    # Engine rejects invalid features
    mock_engine.add_landmarks.side_effect = ValueError("Invalid landmark features")
    
    with pytest.raises(ValueError, match="Invalid landmark features"):
        service.add_landmarks("not a vector")


def test_model_loading_failure() -> None:
    """Test 10: Model loading failure."""
    mock_ms = MagicMock(spec=ModelService)
    # ModelService raises FileNotFoundError when engine cannot be loaded
    mock_ms.get_inference_engine.side_effect = FileNotFoundError("Model file not found")
    
    service = PredictionService(model_service=mock_ms)
    
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        service.add_landmarks(np.random.rand(126))


def test_json_compatibility(mock_model_service: MagicMock, mock_engine: MagicMock) -> None:
    """Test 11: JSON compatibility."""
    service = PredictionService(model_service=mock_model_service)
    mock_engine.is_ready.return_value = True
    
    raw_prediction = {
        "predicted_index": 2,
        "predicted_class": "thank_you",
        "confidence": 0.85,
        "probabilities": [0.05, 0.10, 0.85]
    }
    mock_engine.predict.return_value = raw_prediction
    
    result = service.predict()
    
    try:
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert decoded == result
    except TypeError as e:
        pytest.fail(f"Prediction result is not JSON serializable: {e}")


def test_real_silentvoice_integration() -> None:
    """Test 12: Real SilentVoice integration."""
    model_path = Path("models/checkpoints/best_model.keras")
    metadata_path = Path("datasets/landmarks/metadata.json")
    test_data_path = Path("datasets/landmarks/test.npz")
    
    if not model_path.exists() or not metadata_path.exists() or not test_data_path.exists():
        pytest.skip("Required model, metadata, or test data files are missing")
        
    model_service = ModelService(
        model_path=model_path,
        metadata_path=metadata_path
    )
    
    service = PredictionService(model_service=model_service)
    
    test_data = np.load(str(test_data_path))
    # Pick a random sequence (assume 'X' contains shape (N, 32, 126) or similar)
    # Test data format: X might be (num_samples, sequence_length, features)
    X = test_data["X"]
    if len(X) == 0:
        pytest.skip("Test dataset is empty")
        
    sequence = X[0]
    if sequence.shape != (32, 126):
        pytest.skip(f"Expected sequence shape (32, 126), got {sequence.shape}")
        
    for frame in sequence:
        state = service.add_landmarks(frame)
        
    assert state["sequence_ready"] is True
    assert service.is_ready() is True
    
    result = service.predict()
    
    assert "predicted_class" in result
    assert "predicted_index" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "sequence_ready" in result
    assert "stable" in result
    
    assert result["sequence_ready"] is True
    assert isinstance(result["stable"], bool)
    assert len(result["probabilities"]) > 1  # Should match number of classes
    
    try:
        json.dumps(result)
    except TypeError as e:
        pytest.fail(f"Real prediction result is not JSON serializable: {e}")
