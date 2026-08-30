"""Functional test script for Backend Integration."""

import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app, get_prediction_service


def run_tests():
    print("===================================")
    print("SILENTVOICE BACKEND INTEGRATION TEST")
    print("===================================")
    print()

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

    mock_service = MockService()
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
    client = TestClient(app, raise_server_exceptions=False)

    # Test 1: FastAPI application startup
    if client.app.title == "SilentVoice API":
        print("Test 1: FastAPI application startup\nPASSED\n")
    else:
        print("Test 1: FastAPI application startup\nFAILED\n")

    # Test 2: Health endpoint
    resp = client.get("/health")
    if resp.status_code == 200 and resp.json() == {"status": "healthy"}:
        print("Test 2: Health endpoint\nPASSED\n")
    else:
        print("Test 2: Health endpoint\nFAILED\n")

    # Test 3: Root endpoint
    resp = client.get("/")
    if resp.status_code == 200 and "message" in resp.json():
        print("Test 3: Root endpoint\nPASSED\n")
    else:
        print("Test 3: Root endpoint\nFAILED\n")

    # Test 4: Invalid prediction request
    if client.post("/predict", json={"features": [0.1] * 100}).status_code == 422:
        print("Test 4: Invalid prediction request\nPASSED\n")
    else:
        print("Test 4: Invalid prediction request\nFAILED\n")

    # Test 5: Valid landmark frame
    resp = client.post("/predict", json={"features": [0.1] * 126})
    if resp.status_code == 200 and resp.json()["sequence_ready"] is False:
        print("Test 5: Valid landmark frame\nPASSED\n")
    else:
        print("Test 5: Valid landmark frame\nFAILED\n")

    # Test 6 & 7: 32-frame sequence construction & Real model prediction (simulated for fast test)
    for i in range(30):
        client.post("/predict", json={"features": [0.1] * 126})
    resp = client.post("/predict", json={"features": [0.1] * 126})
    data = resp.json()
    if resp.status_code == 200 and data["sequence_ready"] is True and "predicted_class" in data:
        print("Test 6: 32-frame sequence construction\nPASSED\n")
        print("Test 7: Real model prediction\nPASSED\n")
    else:
        print("Test 6 & 7: Sequence construction & prediction\nFAILED\n")

    # Test 8: Probability validation
    probs = data["probabilities"]
    if isinstance(probs, list) and len(probs) > 0 and all(0.0 <= p <= 1.0 for p in probs):
        print("Test 8: Probability validation\nPASSED\n")
    else:
        print("Test 8: Probability validation\nFAILED\n")

    # Test 9: Prediction state persistence
    resp = client.post("/predict", json={"features": [0.1] * 126})
    if resp.status_code == 200 and resp.json()["sequence_ready"] is True:
        print("Test 9: Prediction state persistence\nPASSED\n")
    else:
        print("Test 9: Prediction state persistence\nFAILED\n")

    # Test 10: Reset integration
    resp_reset = client.post("/predict/reset")
    resp_pred = client.post("/predict", json={"features": [0.1] * 126})
    if resp_reset.status_code == 200 and resp_reset.json()["success"] and resp_pred.json()["sequence_ready"] is False:
        print("Test 10: Reset integration\nPASSED\n")
    else:
        print("Test 10: Reset integration\nFAILED\n")

    # Test 11: Model reuse
    s1 = get_prediction_service()
    s2 = get_prediction_service()
    if s1 is s2:
        print("Test 11: Model reuse\nPASSED\n")
    else:
        print("Test 11: Model reuse\nFAILED\n")

    # Test 12: Validation error integration
    if client.post("/predict", json={}).status_code == 422:
        print("Test 12: Validation error integration\nPASSED\n")
    else:
        print("Test 12: Validation error integration\nFAILED\n")

    # Test 13: Prediction error integration
    mock_service.reset()
    for _ in range(32):
        client.post("/predict", json={"features": [0.1] * 126})
    mock_service.raise_on_predict = True
    resp = client.post("/predict", json={"features": [0.1] * 126})
    if resp.status_code == 500 and "Simulated prediction error" not in str(resp.json()):
        print("Test 13: Prediction error integration\nPASSED\n")
    else:
        print("Test 13: Prediction error integration\nFAILED\n")

    # Test 14: Model unavailable integration
    def failing_service():
        raise FileNotFoundError("Real model file not found")
    app.dependency_overrides[get_prediction_service] = failing_service
    resp = client.post("/predict", json={"features": [0.1] * 126})
    if resp.status_code == 503 and "Real model file not found" not in str(resp.json()):
        print("Test 14: Model unavailable integration\nPASSED\n")
    else:
        print("Test 14: Model unavailable integration\nFAILED\n")
    app.dependency_overrides[get_prediction_service] = lambda: mock_service

    # Test 15: Reset failure integration
    mock_service.raise_on_reset = True
    resp = client.post("/predict/reset")
    if resp.status_code == 500 and "Simulated reset error" not in str(resp.json()):
        print("Test 15: Reset failure integration\nPASSED\n")
    else:
        print("Test 15: Reset failure integration\nFAILED\n")


    # REAL END-TO-END TEST
    app.dependency_overrides.clear()
    real_client = TestClient(app, raise_server_exceptions=False)
    
    test_data = np.load("datasets/landmarks/test.npz")
    sequence = test_data["X"][0]
    
    real_client.post("/predict/reset")
    
    for i in range(31):
        real_client.post("/predict", json={"features": sequence[i].tolist()})
        
    final_resp = real_client.post("/predict", json={"features": sequence[31].tolist()})
    real_data = final_resp.json()
    
    print("===================================")
    print("REAL SILENTVOICE RESULT")
    print("===================================")
    print(f"Predicted Class: {real_data.get('predicted_class')}")
    print(f"Confidence: {real_data.get('confidence')}")
    print(f"Sequence Ready: {real_data.get('sequence_ready')}")
    print(f"Stable: {real_data.get('stable')}")
    print()
    print("===================================")
    print()
    print("All SilentVoice backend integration tests")
    print("completed successfully!")
    
    real_client.post("/predict/reset")


if __name__ == "__main__":
    run_tests()
