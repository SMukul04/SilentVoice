"""Functional test script for the SilentVoice API schemas."""

import math
from pydantic import ValidationError

from backend.app.schemas import PredictionRequest, PredictionResponse

def run_tests() -> None:
    print("===================================")
    print("API SCHEMA TEST")
    print("===================================")
    print()

    # Test 1: Valid PredictionRequest
    try:
        req = PredictionRequest(features=[0.1] * 126)
        print("Test 1: Valid PredictionRequest\nPASSED\n")
    except Exception as e:
        print(f"Test 1: Valid PredictionRequest\nFAILED: {e}\n")

    # Test 2: Invalid feature count
    try:
        PredictionRequest(features=[0.1] * 125)
        print("Test 2: Invalid feature count\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 2: Invalid feature count\nPASSED\n")

    # Test 3: Invalid feature dimensions
    try:
        PredictionRequest(features=[[0.1]] * 126) # type: ignore
        print("Test 3: Invalid feature dimensions\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 3: Invalid feature dimensions\nPASSED\n")

    # Test 4: Invalid numeric values
    try:
        PredictionRequest(features=[0.1] * 125 + [float('nan')])
        print("Test 4: Invalid numeric values\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 4: Invalid numeric values\nPASSED\n")

    # Test 5: Invalid feature types
    try:
        PredictionRequest(features=[0.1] * 125 + ["string"]) # type: ignore
        print("Test 5: Invalid feature types\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 5: Invalid feature types\nPASSED\n")

    # Test 6: Valid PredictionResponse
    try:
        resp = PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=0.99,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )
        print("Test 6: Valid PredictionResponse\nPASSED\n")
    except Exception as e:
        print(f"Test 6: Valid PredictionResponse\nFAILED: {e}\n")

    # Test 7: Invalid confidence
    try:
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=-0.1,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )
        print("Test 7: Invalid confidence\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 7: Invalid confidence\nPASSED\n")

    # Test 8: Invalid probabilities
    try:
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=0.99,
            probabilities=[-0.1, 0.99],
            sequence_ready=True,
            stable=True
        )
        print("Test 8: Invalid probabilities\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 8: Invalid probabilities\nPASSED\n")

    # Test 9: Invalid predicted class
    try:
        PredictionResponse(
            predicted_index=1,
            predicted_class="",
            confidence=0.99,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )
        print("Test 9: Invalid predicted class\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 9: Invalid predicted class\nPASSED\n")

    # Test 10: JSON serialization
    try:
        resp = PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            confidence=0.99,
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        )
        json_str = resp.model_dump_json()
        PredictionResponse.model_validate_json(json_str)
        print("Test 10: JSON serialization\nPASSED\n")
    except Exception as e:
        print(f"Test 10: JSON serialization\nFAILED: {e}\n")

    # Test 11: Schema compatibility with PredictionService output
    try:
        service_output = {
            "predicted_index": 0,
            "predicted_class": "alive",
            "confidence": 0.94,
            "probabilities": [0.94, 0.06],
            "sequence_ready": True,
            "stable": True
        }
        PredictionResponse(**service_output)
        print("Test 11: Schema compatibility with PredictionService output\nPASSED\n")
    except Exception as e:
        print(f"Test 11: Schema compatibility with PredictionService output\nFAILED: {e}\n")

    # Test 12: Invalid response structure
    try:
        PredictionResponse(
            predicted_index=1,
            predicted_class="hello",
            # missing confidence
            probabilities=[0.01, 0.99],
            sequence_ready=True,
            stable=True
        ) # type: ignore
        print("Test 12: Invalid response structure\nFAILED (Did not raise)\n")
    except ValidationError:
        print("Test 12: Invalid response structure\nPASSED\n")

    print("===================================")
    print()
    print("All API schema tests completed successfully!")


if __name__ == "__main__":
    run_tests()
