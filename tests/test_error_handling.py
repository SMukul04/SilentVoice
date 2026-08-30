"""Tests for the error-handling layer of the SilentVoice API."""

import pytest
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app, get_prediction_service
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService


@pytest.fixture
def mock_prediction_service():
    """Provides a mocked PredictionService for fast deterministic error tests."""
    class MockService:
        def __init__(self):
            self.sequence_count = 0
            self.ready = False
            self.raise_on_predict = False
            self.raise_on_reset = False
            
        def add_landmarks(self, features):
            self.sequence_count += 1
            if self.sequence_count >= 32:
                self.ready = True
            return {"sequence_ready": self.ready, "sequence_length": self.sequence_count}
            
        def predict(self):
            if self.raise_on_predict:
                raise RuntimeError("Simulated model prediction error")
            return {
                "predicted_index": 1,
                "predicted_class": "hello",
                "confidence": 0.95,
                "probabilities": [0.05, 0.95],
                "sequence_ready": True,
                "stable": True
            }
            
        def reset(self):
            if self.raise_on_reset:
                raise Exception("Simulated reset failure")
            self.sequence_count = 0
            self.ready = False
            
    return MockService()


@pytest.fixture
def client(mock_prediction_service):
    """Provides a FastAPI TestClient with the mocked service injected."""
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_still_works(client: TestClient) -> None:
    """Test 1: Normal health endpoint still works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_request_body(client: TestClient) -> None:
    """Test 2: Invalid request body returns HTTP 422."""
    response = client.post("/predict", json={"wrong_key": "data"})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Validation error"


def test_missing_features(client: TestClient) -> None:
    """Test 3: Missing features returns HTTP 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Validation error"


def test_wrong_feature_count(client: TestClient) -> None:
    """Test 4: Wrong feature count returns HTTP 422."""
    response = client.post("/predict", json={"features": [0.1] * 125})
    assert response.status_code == 422
    assert response.json()["error"] == "Validation error"


def test_invalid_numeric_values(client: TestClient) -> None:
    """Test 5: NaN/infinity input returns HTTP 422."""
    # Use string representations to avoid httpx JSON serialization crash,
    # Pydantic will cast them to float and then our validator will reject them.
    response = client.post("/predict", json={"features": ["NaN"] * 126})
    assert response.status_code == 422
    assert response.json()["error"] == "Validation error"
    
    response = client.post("/predict", json={"features": ["Infinity"] * 126})
    assert response.status_code == 422


def test_model_loading_failure(client: TestClient, mock_prediction_service) -> None:
    """Test 6: Model loading failure returns HTTP 503."""
    # To test model loading failure, we override the dependency to raise FileNotFoundError
    def failing_service():
        raise FileNotFoundError("Simulated model missing")
        
    app.dependency_overrides[get_prediction_service] = failing_service
    response = client.post("/predict", json={"features": [0.1] * 126})
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    
    assert response.status_code == 503
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Model unavailable"
    assert "Simulated model missing" not in data["detail"]


def test_prediction_failure(client: TestClient, mock_prediction_service) -> None:
    """Test 7: Prediction failure returns HTTP 500."""
    mock_prediction_service.raise_on_predict = True
    
    for _ in range(32):
        response = client.post("/predict", json={"features": [0.1] * 126})
        
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Prediction failed"
    assert "Simulated model prediction error" not in data["detail"]


def test_unexpected_exception(client: TestClient) -> None:
    """Test 8: Unexpected exception returns HTTP 500."""
    def unexpectedly_failing_service():
        raise ValueError("Something really weird happened")
        
    app.dependency_overrides[get_prediction_service] = unexpectedly_failing_service
    response = client.post("/predict", json={"features": [0.1] * 126})
    
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Prediction failed"
    assert "Something really weird happened" not in data["detail"]


def test_error_json_structure(client: TestClient) -> None:
    """Test 9: Error responses have the expected JSON structure."""
    response = client.post("/predict", json={})
    data = response.json()
    assert "success" in data
    assert "error" in data
    assert "detail" in data
    assert len(data.keys()) == 3


def test_no_stack_traces(client: TestClient, mock_prediction_service) -> None:
    """Test 10 & 11: Error responses do not expose Python stack traces or file paths."""
    mock_prediction_service.raise_on_predict = True
    for _ in range(32):
        response = client.post("/predict", json={"features": [0.1] * 126})
        
    data = response.json()
    response_str = str(data).lower()
    
    # Ensure no common stack trace indicators or file paths are present
    assert "traceback" not in response_str
    assert "file" not in response_str
    assert "line" not in response_str
    assert "/" not in response_str
    assert "\\" not in response_str


def test_reset_failure(client: TestClient, mock_prediction_service) -> None:
    """Test 12: Reset failure returns HTTP 500."""
    mock_prediction_service.raise_on_reset = True
    response = client.post("/predict/reset")
    
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "Prediction failed"


def test_successful_prediction_behavior(client: TestClient, mock_prediction_service) -> None:
    """Test 13: Successful prediction behavior remains unchanged."""
    mock_prediction_service.reset()
    for _ in range(31):
        response = client.post("/predict", json={"features": [0.1] * 126})
        assert response.status_code == 200
        assert response.json()["sequence_ready"] is False
        
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    assert response.json()["sequence_ready"] is True
    assert response.json()["predicted_class"] == "hello"


def test_real_silentvoice_integration() -> None:
    """Test 14: Real SilentVoice integration."""
    app.dependency_overrides.clear()
    real_client = TestClient(app)
    
    model_path = Path("models/checkpoints/best_model.keras")
    metadata_path = Path("datasets/landmarks/metadata.json")
    test_data_path = Path("datasets/landmarks/test.npz")
    
    if not (model_path.exists() and metadata_path.exists() and test_data_path.exists()):
        pytest.skip("Required model, metadata, or test data files are missing")
        
    real_client.post("/predict/reset")
    
    test_data = np.load(str(test_data_path))
    X = test_data["X"]
    sequence = X[0]
    
    for i in range(31):
        response = real_client.post("/predict", json={"features": sequence[i].tolist()})
        assert response.status_code == 200
        
    response = real_client.post("/predict", json={"features": sequence[31].tolist()})
    assert response.status_code == 200
    
    data = response.json()
    assert data["sequence_ready"] is True
    assert "predicted_class" in data
    assert 0.0 <= data["confidence"] <= 1.0
