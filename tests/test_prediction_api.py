"""Unit tests for the Prediction API Endpoint."""

from pathlib import Path
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, get_prediction_service
from backend.services.prediction_service import PredictionService
from backend.services.model_service import ModelService
from backend.realtime.prediction_stabilizer import PredictionStabilizer


@pytest.fixture
def mock_prediction_service():
    """Provides a mocked PredictionService for fast deterministic tests."""
    class MockService:
        def __init__(self):
            self.sequence_count = 0
            self.ready = False
            self.raise_on_predict = False

        def add_landmarks(self, features):
            self.sequence_count += 1
            if self.sequence_count >= 32:
                self.ready = True
            return {"sequence_ready": self.ready, "sequence_length": self.sequence_count}

        def predict(self):
            if self.raise_on_predict:
                raise RuntimeError("Simulated model error")
            return {
                "predicted_index": 1,
                "predicted_class": "hello",
                "confidence": 0.95,
                "probabilities": [0.05, 0.95],
                "sequence_ready": True,
                "stable": True
            }

        def reset(self):
            self.sequence_count = 0
            self.ready = False

    return MockService()


@pytest.fixture
def client(mock_prediction_service):
    """Provides a FastAPI TestClient with the mocked service injected."""
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_app_initialization(client: TestClient) -> None:
    """Test 1: FastAPI application initialization."""
    assert client.app.title == "SilentVoice API"


def test_health_endpoint(client: TestClient) -> None:
    """Test 2: GET /health still works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_feature_count(client: TestClient) -> None:
    """Test 3: POST /predict with invalid feature count."""
    response = client.post("/predict", json={"features": [0.1] * 125})
    assert response.status_code == 422


def test_invalid_feature_values(client: TestClient) -> None:
    """Test 4: POST /predict with invalid feature values."""
    # NaN
    response = client.post("/predict", json={"features": ["NaN"] * 126})
    assert response.status_code == 422

    # string
    response = client.post("/predict", json={"features": ["str"] * 126})
    assert response.status_code == 422


def test_prediction_before_ready(client: TestClient) -> None:
    """Test 5: POST /predict with valid input before sequence is ready."""
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_ready"] is False
    assert data["predicted_class"] == "unknown"


def test_sequence_filling_and_prediction(client: TestClient) -> None:
    """Test 6 & 8: Send enough requests to fill sequence and verify state persistence."""
    for i in range(31):
        response = client.post("/predict", json={"features": [0.1] * 126})
        assert response.status_code == 200
        assert response.json()["sequence_ready"] is False

    # The 32nd request should trigger a prediction
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_ready"] is True
    assert data["predicted_class"] == "hello"


def test_valid_prediction_response_structure(client: TestClient) -> None:
    """Test 7: Verify valid PredictionResponse structure."""
    # Fast-forward to 31 frames
    for _ in range(31):
        client.post("/predict", json={"features": [0.1] * 126})

    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    data = response.json()

    assert "predicted_index" in data
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "sequence_ready" in data
    assert "stable" in data


def test_reset_endpoint(client: TestClient) -> None:
    """Test 9: POST /predict/reset."""
    client.post("/predict", json={"features": [0.1] * 126})

    response = client.post("/predict/reset")
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Prediction state reset"}

    # Verify sequence was reset by sending another request
    response2 = client.post("/predict", json={"features": [0.1] * 126})
    assert response2.json()["sequence_ready"] is False


def test_prediction_error(mock_prediction_service) -> None:
    """Test 10: Verify prediction/model errors return HTTP 500."""
    mock_prediction_service.raise_on_predict = True

    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    with TestClient(app, raise_server_exceptions=False) as client:
        # Fast-forward to ready state
        for _ in range(32):
            response = client.post("/predict", json={"features": [0.1] * 126})

    assert response.status_code == 500
    assert "detail" in response.json()
    app.dependency_overrides.clear()


def test_model_service_is_reused() -> None:
    """Test 11: Verify ModelService is reused (via dependency injection defaults)."""
    # Simply getting the default dependency should return the same singleton
    service1 = get_prediction_service()
    service2 = get_prediction_service()
    assert service1 is service2
    assert service1.model_service is service2.model_service


def test_real_silentvoice_integration() -> None:
    """Test 12: Real SilentVoice integration."""
    model_path = Path("models/checkpoints/best_model.keras")
    metadata_path = Path("datasets/landmarks/metadata.json")
    test_data_path = Path("datasets/landmarks/test.npz")

    if not (model_path.exists() and metadata_path.exists() and test_data_path.exists()):
        pytest.skip("Required model, metadata, or test data files are missing")

    # We will use the actual app without overriding dependencies
    # But since it's a singleton, let's reset it first just in case
    app.dependency_overrides.clear()
    real_client = TestClient(app)

    real_client.post("/predict/reset")

    test_data = np.load(str(test_data_path))
    X = test_data["X"]
    if len(X) == 0:
        pytest.skip("Test dataset is empty")

    sequence = X[0]

    # Send 31 frames
    for i in range(31):
        payload = {"features": sequence[i].tolist()}
        response = real_client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["sequence_ready"] is False

    # Send the 32nd frame
    payload = {"features": sequence[31].tolist()}
    response = real_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sequence_ready"] is True
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) > 1
