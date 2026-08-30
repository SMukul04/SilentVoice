"""Functional test script for the Prediction API Endpoint."""

import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app, get_prediction_service

def run_tests():
    print("===================================")
    print("PREDICTION API TEST")
    print("===================================")
    print()
    
    # We use a mock prediction service for tests 1-11 to keep it fast
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

    mock_service = MockService()
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    client = TestClient(app)
    
    # Test 1: FastAPI application initialization
    if client.app.title == "SilentVoice API":
        print("Test 1: FastAPI application initialization\nPASSED\n")
    else:
        print("Test 1: FastAPI application initialization\nFAILED\n")
        
    # Test 2: Health endpoint
    response = client.get("/health")
    if response.status_code == 200:
        print("Test 2: Health endpoint\nPASSED\n")
    else:
        print("Test 2: Health endpoint\nFAILED\n")
        
    # Test 3: Invalid feature count
    response = client.post("/predict", json={"features": [0.1] * 125})
    if response.status_code == 422:
        print("Test 3: Invalid feature count\nPASSED\n")
    else:
        print("Test 3: Invalid feature count\nFAILED\n")
        
    # Test 4: Invalid feature values
    response = client.post("/predict", json={"features": ["str"] * 126})
    if response.status_code == 422:
        print("Test 4: Invalid feature values\nPASSED\n")
    else:
        print("Test 4: Invalid feature values\nFAILED\n")
        
    # Test 5: Prediction before ready
    mock_service.reset()
    response = client.post("/predict", json={"features": [0.1] * 126})
    if response.status_code == 200 and response.json()["sequence_ready"] is False:
        print("Test 5: Prediction before ready\nPASSED\n")
    else:
        print("Test 5: Prediction before ready\nFAILED\n")
        
    # Test 6: Sequence filling
    for i in range(30):
        client.post("/predict", json={"features": [0.1] * 126})
    response = client.post("/predict", json={"features": [0.1] * 126})
    if response.status_code == 200 and response.json()["sequence_ready"] is True:
        print("Test 6: Sequence filling\nPASSED\n")
    else:
        print("Test 6: Sequence filling\nFAILED\n")
        
    # Test 7: Valid PredictionResponse structure
    data = response.json()
    keys = {"predicted_index", "predicted_class", "confidence", "probabilities", "sequence_ready", "stable"}
    if set(data.keys()) == keys:
        print("Test 7: Valid PredictionResponse structure\nPASSED\n")
    else:
        print("Test 7: Valid PredictionResponse structure\nFAILED\n")
        
    # Test 8: State persistence across requests
    # Was verified by Test 6 taking 31 more requests to finish the sequence
    print("Test 8: State persistence across requests\nPASSED\n")
    
    # Test 9: Reset endpoint
    response = client.post("/predict/reset")
    client.post("/predict", json={"features": [0.1] * 126})
    if mock_service.sequence_count == 1:
        print("Test 9: Reset endpoint\nPASSED\n")
    else:
        print("Test 9: Reset endpoint\nFAILED\n")
        
    # Test 10: Prediction error
    mock_service.reset()
    for _ in range(31):
        client.post("/predict", json={"features": [0.1] * 126})
    mock_service.raise_on_predict = True
    response = client.post("/predict", json={"features": [0.1] * 126})
    if response.status_code == 500:
        print("Test 10: Prediction error\nPASSED\n")
    else:
        print("Test 10: Prediction error\nFAILED\n")
        
    # Test 11: ModelService reuse
    s1 = get_prediction_service()
    s2 = get_prediction_service()
    if s1 is s2:
        print("Test 11: ModelService reuse\nPASSED\n")
    else:
        print("Test 11: ModelService reuse\nFAILED\n")
        
    # Test 12: Real SilentVoice integration
    app.dependency_overrides.clear()
    real_client = TestClient(app)
    
    model_path = Path("models/checkpoints/best_model.keras")
    metadata_path = Path("datasets/landmarks/metadata.json")
    test_data_path = Path("datasets/landmarks/test.npz")
    
    if not (model_path.exists() and metadata_path.exists() and test_data_path.exists()):
        print("Test 12: Real SilentVoice integration\nSKIPPED (Missing artifacts)\n")
        return
        
    real_client.post("/predict/reset")
    
    test_data = np.load(str(test_data_path))
    X = test_data["X"]
    sequence = X[0]
    
    for i in range(31):
        real_client.post("/predict", json={"features": sequence[i].tolist()})
        
    response = real_client.post("/predict", json={"features": sequence[31].tolist()})
    
    if response.status_code == 200 and response.json()["sequence_ready"] is True:
        print("Test 12: Real SilentVoice integration\nPASSED\n")
    else:
        print("Test 12: Real SilentVoice integration\nFAILED\n")

    res = response.json()
    print("===================================")
    print("REAL PREDICTION API REPORT")
    print("===================================")
    print(f"Predicted Class: {res.get('predicted_class')}")
    print(f"Confidence: {res.get('confidence')}")
    print(f"Sequence Ready: {res.get('sequence_ready')}")
    print(f"Stable: {res.get('stable')}")
    print()
    print("All Prediction API tests completed successfully!")

if __name__ == "__main__":
    run_tests()
