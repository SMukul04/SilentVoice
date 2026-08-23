"""Runnable verification script for the SilentVoice InferenceEngine."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.inference_engine import InferenceEngine


MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.keras"
METADATA_PATH = PROJECT_ROOT / "datasets" / "landmarks" / "metadata.json"
TEST_DATA_PATH = PROJECT_ROOT / "datasets" / "landmarks" / "test.npz"


def run_test(name: str, test: Callable[[], None]) -> None:
    """Run one assertion-based check and print its required status line."""
    print(f"\n{name}")
    test()
    print("PASSED")


def assert_valid_prediction(result: dict[str, object], engine: InferenceEngine) -> None:
    """Assert the public prediction contract returned by the inference engine."""
    required_keys = {"predicted_index", "predicted_class", "confidence", "probabilities"}
    assert required_keys == set(result), "Prediction result has unexpected keys"
    assert isinstance(result["predicted_index"], int)
    assert isinstance(result["predicted_class"], str)
    assert isinstance(result["confidence"], float)
    assert np.isfinite(result["confidence"])
    assert isinstance(result["probabilities"], list)
    assert len(result["probabilities"]) == engine.num_classes
    assert all(isinstance(value, float) and np.isfinite(value) for value in result["probabilities"])


def main() -> None:
    """Run all Module 6.1 checks, including the local trained-model integration test."""
    for required_path in (MODEL_PATH, METADATA_PATH, TEST_DATA_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required SilentVoice artifact is missing: {required_path}")

    print("===================================")
    print("INFERENCE ENGINE TEST")
    print("===================================")

    engine = InferenceEngine(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    real_sequence: np.ndarray | None = None
    real_label: int | None = None
    real_result: dict[str, object] | None = None

    def component_initialization() -> None:
        fresh_engine = InferenceEngine()
        assert fresh_engine.sequence_length == 32
        assert fresh_engine.feature_dimension == 126
        assert fresh_engine.confidence_threshold == 0.0
        assert fresh_engine.sequence_count == 0

    def model_and_metadata_loading() -> None:
        engine.load()
        assert engine.model is not None
        assert engine.index_to_class is not None
        assert engine.num_classes == 13
        assert engine.model.input_shape == (None, 32, 126)
        assert engine.model.output_shape == (None, 13)

    def invalid_landmark_input() -> None:
        invalid_vectors = (
            None,
            np.zeros(125, dtype=np.float32),
            np.zeros((1, 126), dtype=np.float32),
            ["invalid"] * 126,
            np.full(126, np.nan, dtype=np.float32),
            np.full(126, np.inf, dtype=np.float32),
        )
        for invalid_vector in invalid_vectors:
            try:
                engine.add_landmarks(invalid_vector)
            except (TypeError, ValueError) as error:
                assert str(error), "Invalid input error must be clear"
            else:
                raise AssertionError("Invalid landmark input was accepted")

    def sequence_buffer_behavior() -> None:
        engine.reset()
        for value in range(33):
            engine.add_landmarks(np.full(126, value, dtype=np.float32))
        assert engine.sequence_count == 32
        assert np.array_equal(engine._buffer[0], np.full(126, 1, dtype=np.float32))
        assert np.array_equal(engine._buffer[-1], np.full(126, 32, dtype=np.float32))

    def readiness_behavior() -> None:
        engine.reset()
        vector = np.zeros(126, dtype=np.float32)
        for _ in range(31):
            engine.add_landmarks(vector)
        assert not engine.is_ready()
        engine.add_landmarks(vector)
        assert engine.is_ready()

    def reset_behavior() -> None:
        engine.reset()
        assert engine.sequence_count == 0
        assert not engine.is_ready()

    def prediction_before_ready() -> None:
        try:
            engine.predict()
        except RuntimeError as error:
            assert "not ready" in str(error).lower()
        else:
            raise AssertionError("Prediction succeeded before 32 frames were available")

    def prediction_with_synthetic_sequence() -> None:
        engine.reset()
        sequence = np.linspace(0.0, 1.0, 32 * 126, dtype=np.float32).reshape(32, 126)
        for frame in sequence:
            engine.add_landmarks(frame)
        assert_valid_prediction(engine.predict(), engine)

    def confidence_threshold_behavior() -> None:
        threshold_engine = InferenceEngine(
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
            confidence_threshold=1.1,
        )
        threshold_engine.load()
        for frame in np.zeros((32, 126), dtype=np.float32):
            threshold_engine.add_landmarks(frame)
        result = threshold_engine.predict()
        assert_valid_prediction(result, threshold_engine)
        assert result["predicted_class"] == "unknown"

    def real_silentvoice_dataset_integration() -> None:
        nonlocal real_sequence, real_label, real_result
        with np.load(TEST_DATA_PATH) as dataset:
            real_sequence = dataset["X"][0].astype(np.float32, copy=True)
            real_label = int(dataset["y"][0])
        assert real_sequence.shape == (32, 126)
        engine.reset()
        for frame in real_sequence:
            engine.add_landmarks(frame)
        real_result = engine.predict()
        assert_valid_prediction(real_result, engine)

    run_test("Test 1: Component initialization", component_initialization)
    run_test("Test 2: Model and metadata loading", model_and_metadata_loading)
    run_test("Test 3: Invalid landmark input", invalid_landmark_input)
    run_test("Test 4: Sequence buffer behavior", sequence_buffer_behavior)
    run_test("Test 5: Readiness behavior", readiness_behavior)
    run_test("Test 6: Reset behavior", reset_behavior)
    run_test("Test 7: Prediction before ready", prediction_before_ready)
    run_test("Test 8: Prediction with synthetic sequence", prediction_with_synthetic_sequence)
    run_test("Test 9: Confidence threshold behavior", confidence_threshold_behavior)
    run_test("Test 10: Real SilentVoice dataset integration", real_silentvoice_dataset_integration)

    assert real_result is not None and real_label is not None and engine.index_to_class is not None
    true_class = engine.index_to_class[real_label]
    predicted_class = real_result["predicted_class"]
    correct = real_result["predicted_index"] == real_label

    print("\n===================================")
    print("REAL INFERENCE REPORT")
    print("===================================")
    print(f"True Class Index: {real_label}")
    print(f"True Class: {true_class}")
    print(f"Predicted Class Index: {real_result['predicted_index']}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {real_result['confidence']:.6f}")
    print(f"Correct: {correct}")
    print("\nAll InferenceEngine tests completed successfully!")


if __name__ == "__main__":
    main()
