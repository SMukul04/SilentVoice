"""Full Backend Integration Testing for SilentVoice API."""

import pytest
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app, get_prediction_service


@pytest.fixture
def mock_prediction_service():
    """Provides a mocked PredictionService for fast deterministic error tests."""
    class MockService:
        def __init__(self):
            self.sequence_count = 0
            self.ready = False
            self.raise_on_predict = False
            self.raise_on_reset = False
            self.model_service = self
            
        def add_landmarks(self, features):
            self.sequence_count += 1
            if self.sequence_count >= 32:
                self.ready = True
            return {"sequence_ready": self.ready, "sequence_length": self.sequence_count}
            
        def predict(self):
            if self.raise_on_predict:
                raise RuntimeError("Simulated prediction error")
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
                raise Exception("Simulated reset error")
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


def test_fastapi_app_starts(client: TestClient) -> None:
    """Test 1: FastAPI application starts successfully."""
    assert client.app.title == "SilentVoice API"


def test_health_endpoint(client: TestClient) -> None:
    """Test 2: GET /health returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client: TestClient) -> None:
    """Test 3: Root endpoint works."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to SilentVoice API"}


def test_invalid_prediction_request(client: TestClient) -> None:
    """Test 4: Invalid prediction request returns 422."""
    response = client.post("/predict", json={"features": [0.1] * 100})
    assert response.status_code == 422


def test_valid_landmark_frame_accepted(client: TestClient) -> None:
    """Test 5: Valid landmark frame is accepted, returns sequence_ready=false."""
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    assert response.json()["sequence_ready"] is False


def test_32_frame_sequence_construction_and_prediction(client: TestClient) -> None:
    """Test 6 & 7: Construct 32-frame sequence and verify real prediction structure."""
    for i in range(31):
        response = client.post("/predict", json={"features": [0.1] * 126})
        assert response.status_code == 200
        assert response.json()["sequence_ready"] is False
        
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    
    data = response.json()
    assert data["sequence_ready"] is True
    assert "predicted_index" in data
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "stable" in data
    
    assert 0.0 <= data["confidence"] <= 1.0


def test_probability_validation(client: TestClient) -> None:
    """Test 8: Probability validation."""
    for i in range(32):
        response = client.post("/predict", json={"features": [0.1] * 126})
    
    data = response.json()
    probs = data["probabilities"]
    
    assert isinstance(probs, list)
    assert len(probs) > 0
    for p in probs:
        assert isinstance(p, (float, int))
        assert not np.isnan(p)
        assert not np.isinf(p)
        assert 0.0 <= p <= 1.0


def test_prediction_state_persistence(client: TestClient) -> None:
    """Test 9: Prediction state persistence."""
    for i in range(32):
        client.post("/predict", json={"features": [0.1] * 126})
        
    # Send another frame, it should remain ready without starting from 0
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    assert response.json()["sequence_ready"] is True


def test_reset_integration(client: TestClient) -> None:
    """Test 10: Reset integration."""
    for i in range(32):
        client.post("/predict", json={"features": [0.1] * 126})
        
    response = client.post("/predict/reset")
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 200
    assert response.json()["sequence_ready"] is False


def test_model_reuse() -> None:
    """Test 11: Model reuse verification."""
    # We test that the dependency injects the same service singleton
    s1 = get_prediction_service()
    s2 = get_prediction_service()
    assert s1 is s2
    assert s1.model_service is s2.model_service


def test_validation_error_integration(client: TestClient) -> None:
    """Test 12: Validation error integration."""
    # Missing features
    assert client.post("/predict", json={}).status_code == 422
    # 125 features
    assert client.post("/predict", json={"features": [0.1] * 125}).status_code == 422
    # 127 features
    assert client.post("/predict", json={"features": [0.1] * 127}).status_code == 422
    # Invalid feature type
    assert client.post("/predict", json={"features": ["string"] * 126}).status_code == 422


def test_prediction_error_integration(client: TestClient, mock_prediction_service) -> None:
    """Test 13: Prediction error integration."""
    mock_prediction_service.raise_on_predict = True
    for _ in range(32):
        response = client.post("/predict", json={"features": [0.1] * 126})
        
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert "Simulated prediction error" not in str(data)
    assert "traceback" not in str(data).lower()


def test_model_unavailable_integration(client: TestClient) -> None:
    """Test 14: Model unavailable integration."""
    def failing_service():
        raise FileNotFoundError("Real model file not found")
        
    app.dependency_overrides[get_prediction_service] = failing_service
    response = client.post("/predict", json={"features": [0.1] * 126})
    assert response.status_code == 503
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert "Real model file not found" not in str(data)


def test_reset_failure_integration(client: TestClient, mock_prediction_service) -> None:
    """Test 15: Reset failure integration."""
    mock_prediction_service.raise_on_reset = True
    response = client.post("/predict/reset")
    
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert "Simulated reset error" not in str(data)


def test_real_silentvoice_end_to_end() -> None:
    """REAL END-TO-END TEST."""
    app.dependency_overrides.clear()
    
    with TestClient(app, raise_server_exceptions=False) as real_client:
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
            assert response.json()["sequence_ready"] is False
            
        response = real_client.post("/predict", json={"features": sequence[31].tolist()})
        assert response.status_code == 200
        
        data = response.json()
        assert data["sequence_ready"] is True
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "stable" in data
        assert 0.0 <= data["confidence"] <= 1.0
        
        probs = data["probabilities"]
        assert isinstance(probs, list)
        
        real_client.post("/predict/reset")
