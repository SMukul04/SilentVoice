"""Runnable checks for Module 6.4 PredictionStabilizer."""

from pathlib import Path
import sys
import unittest
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from tests.test_prediction_stabilizer import TestPredictionStabilizer
    labels = ["Component initialization", "Invalid configuration", "Invalid prediction input", "Single prediction behavior", "Probability averaging", "Window size behavior", "Consistency requirement", "Confidence threshold", "Incompatible probability lengths", "Reset behavior", "JSON compatibility", "InferenceEngine integration"]
    methods = [name for name in dir(TestPredictionStabilizer) if name.startswith("test_")]
    methods.sort()
    # Explicit source-order mapping keeps user-facing output stable.
    methods = ["test_initialization", "test_invalid_configuration", "test_invalid_prediction_input", "test_single_prediction", "test_probability_averaging", "test_window_size", "test_consistency", "test_confidence_threshold", "test_incompatible_lengths", "test_reset", "test_json_compatibility", "test_inference_engine_integration"]
    print("===================================\nPREDICTION STABILIZER TEST\n===================================")
    for number, (label, method) in enumerate(zip(labels, methods), 1):
        result = unittest.TextTestRunner(stream=StringIO(), verbosity=0).run(unittest.TestSuite([TestPredictionStabilizer(method)]))
        if not result.wasSuccessful():
            raise RuntimeError(f"Test {number} failed")
        print(f"\nTest {number}: {label}\nPASSED")
    print("\n===================================\n\nAll PredictionStabilizer tests completed successfully!")


if __name__ == "__main__":
    main()
