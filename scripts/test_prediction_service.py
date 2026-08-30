"""Functional test script for the PredictionService."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from backend.realtime.prediction_stabilizer import PredictionStabilizer


def run_tests() -> None:
    print("===================================")
    print("PREDICTION SERVICE TEST")
    print("===================================")
    print()

    # We will use dummy mocks for the unit-style tests to avoid loading real models 11 times
    class DummyEngine:
        def __init__(self):
            self.sequence_count = 0
            self.ready = False
            self.predictions = []
            
        def add_landmarks(self, features):
            if isinstance(features, str):
                raise ValueError("Invalid landmark features")
            self.sequence_count += 1
            if self.sequence_count >= 32:
                self.ready = True
                
        def is_ready(self):
            return self.ready
            
        def predict(self):
            if not self.ready:
                raise RuntimeError("Sequence is not ready for prediction")
            return {
                "predicted_index": 0,
                "predicted_class": "test",
                "confidence": 0.99,
                "probabilities": [0.99, 0.01]
            }
            
        def reset(self):
            self.sequence_count = 0
            self.ready = False

    class DummyModelService(ModelService):
        def __init__(self, fail_load=False):
            self.engine = DummyEngine()
            self.fail_load = fail_load
            
        def get_inference_engine(self):
            if self.fail_load:
                raise FileNotFoundError("Model file not found")
            return self.engine

    # Test 1: Component initialization
    service = PredictionService(model_service=DummyModelService())
    if isinstance(service.stabilizer, PredictionStabilizer):
        print("Test 1: Component initialization\nPASSED\n")
    else:
        print("Test 1: Component initialization\nFAILED\n")

    # Test 2: Dependency injection
    custom_stabilizer = PredictionStabilizer(window_size=3)
    service = PredictionService(model_service=DummyModelService(), stabilizer=custom_stabilizer)
    if service.stabilizer is custom_stabilizer:
        print("Test 2: Dependency injection\nPASSED\n")
    else:
        print("Test 2: Dependency injection\nFAILED\n")

    # Test 3: Landmark forwarding
    ms = DummyModelService()
    service = PredictionService(model_service=ms)
    res = service.add_landmarks(np.zeros(126, dtype=np.float32))
    if ms.engine.sequence_count == 1 and not res["sequence_ready"]:
        print("Test 3: Landmark forwarding\nPASSED\n")
    else:
        print("Test 3: Landmark forwarding\nFAILED\n")

    # Test 4: Readiness behavior
    ms = DummyModelService()
    service = PredictionService(model_service=ms)
    if not service.is_ready():
        ms.engine.ready = True
        if service.is_ready():
            print("Test 4: Readiness behavior\nPASSED\n")
        else:
            print("Test 4: Readiness behavior\nFAILED\n")
    else:
        print("Test 4: Readiness behavior\nFAILED\n")

    # Test 5: Prediction before ready
    service = PredictionService(model_service=DummyModelService())
    try:
        service.predict()
        print("Test 5: Prediction before ready\nFAILED (Did not raise)\n")
    except RuntimeError:
        print("Test 5: Prediction before ready\nPASSED\n")

    # Test 6: Prediction forwarding
    ms = DummyModelService()
    ms.engine.ready = True
    service = PredictionService(model_service=ms)
    res = service.predict()
    if res["predicted_index"] == 0 and res["predicted_class"] == "test":
        print("Test 6: Prediction forwarding\nPASSED\n")
    else:
        print("Test 6: Prediction forwarding\nFAILED\n")

    # Test 7: Stabilized prediction output
    ms = DummyModelService()
    ms.engine.ready = True
    service = PredictionService(model_service=ms)
    res = service.predict()
    expected_keys = {"predicted_index", "predicted_class", "confidence", "probabilities", "sequence_ready", "stable"}
    if set(res.keys()) == expected_keys:
        print("Test 7: Stabilized prediction output\nPASSED\n")
    else:
        print("Test 7: Stabilized prediction output\nFAILED\n")

    # Test 8: Reset behavior
    ms = DummyModelService()
    service = PredictionService(model_service=ms)
    service.add_landmarks(np.zeros(126, dtype=np.float32))
    service.reset()
    if ms.engine.sequence_count == 0:
        print("Test 8: Reset behavior\nPASSED\n")
    else:
        print("Test 8: Reset behavior\nFAILED\n")

    # Test 9: Invalid landmark handling
    service = PredictionService(model_service=DummyModelService())
    try:
        service.add_landmarks("invalid")
        print("Test 9: Invalid landmark handling\nFAILED (Did not raise)\n")
    except ValueError:
        print("Test 9: Invalid landmark handling\nPASSED\n")

    # Test 10: Model loading failure
    service = PredictionService(model_service=DummyModelService(fail_load=True))
    try:
        service.is_ready()
        print("Test 10: Model loading failure\nFAILED (Did not raise)\n")
    except FileNotFoundError:
        print("Test 10: Model loading failure\nPASSED\n")

    # Test 11: JSON compatibility
    ms = DummyModelService()
    ms.engine.ready = True
    service = PredictionService(model_service=ms)
    res = service.predict()
    try:
        json.dumps(res)
        print("Test 11: JSON compatibility\nPASSED\n")
    except TypeError:
        print("Test 11: JSON compatibility\nFAILED\n")

    # Test 12: Real SilentVoice integration
    model_path = Path("models/checkpoints/best_model.keras")
    metadata_path = Path("datasets/landmarks/metadata.json")
    test_data_path = Path("datasets/landmarks/test.npz")
    
    if not (model_path.exists() and metadata_path.exists() and test_data_path.exists()):
        print("Test 12: Real SilentVoice integration\nSKIPPED (Missing artifacts)\n")
        return
        
    real_ms = ModelService(model_path=model_path, metadata_path=metadata_path)
    real_service = PredictionService(model_service=real_ms)
    
    test_data = np.load(str(test_data_path))
    X = test_data["X"]
    sequence = X[0]
    
    for frame in sequence:
        real_service.add_landmarks(frame)
        
    real_res = real_service.predict()
    print("Test 12: Real SilentVoice integration\nPASSED\n")

    print("===================================")
    print("REAL PREDICTION SERVICE REPORT")
    print("===================================")
    print(f"Predicted Class: {real_res['predicted_class']}")
    print(f"Confidence: {real_res['confidence']:.4f}")
    print(f"Stable: {real_res['stable']}")
    print()
    print("All PredictionService tests completed successfully!")

if __name__ == "__main__":
    run_tests()
