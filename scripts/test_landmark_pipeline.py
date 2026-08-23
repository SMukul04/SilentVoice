"""Runnable verification script for Module 6.2 real-time landmark pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.realtime.landmark_pipeline import RealTimeLandmarkPipeline
from backend.sign_recognition.frame_features import FrameFeatures
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.normalizer import LandmarkNormalizer


class MockDetector:
    """Deterministic detector for non-MediaPipe pipeline verification."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"success": True, "num_hands": 1, "landmarks": []}
        self.close_called = False

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        return self.result

    def close(self) -> None:
        self.close_called = True


class MockExtractor:
    """Extractor fake preserving the existing FrameFeatures hand-off convention."""

    def extract(self, detection_result: dict[str, Any]) -> FrameFeatures:
        return FrameFeatures()


class MockNormalizer:
    """Normalizer fake returning controlled model-compatible feature vectors."""

    def __init__(self, output: object | None = None) -> None:
        self.output = np.arange(126, dtype=np.float32) if output is None else output

    def normalize(self, frame_features: FrameFeatures) -> object:
        return self.output


class MockInferenceEngine:
    """Small fake that verifies pipeline-to-inference-engine interaction."""

    def __init__(self, ready_after: int = 32) -> None:
        self.ready_after = ready_after
        self.frames: list[np.ndarray] = []
        self.predict_calls = 0
        self.reset_called = False

    def add_landmarks(self, features: np.ndarray) -> None:
        self.frames.append(features.copy())

    def is_ready(self) -> bool:
        return len(self.frames) >= self.ready_after

    def predict(self) -> dict[str, Any]:
        self.predict_calls += 1
        return {
            "predicted_index": 1,
            "predicted_class": "mock_sign",
            "confidence": 0.95,
            "probabilities": [0.05, 0.95],
        }

    def reset(self) -> None:
        self.frames.clear()
        self.reset_called = True


def run_test(name: str, test: Callable[[], None]) -> None:
    """Run a named assertion test and print the required result format."""
    print(f"\n{name}")
    test()
    print("PASSED")


def make_pipeline(
    output: object | None = None, inference_engine: MockInferenceEngine | None = None
) -> tuple[RealTimeLandmarkPipeline, MockDetector]:
    """Create a pipeline with controlled non-camera dependencies."""
    detector = MockDetector()
    pipeline = RealTimeLandmarkPipeline(
        detector=detector,
        extractor=MockExtractor(),
        normalizer=MockNormalizer(output),
        inference_engine=inference_engine,
    )
    return pipeline, detector


def main() -> None:
    """Run the Module 6.2 checks with mocks and the real landmark components."""
    print("===================================")
    print("REAL-TIME LANDMARK PIPELINE TEST")
    print("===================================")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def component_initialization() -> None:
        pipeline = RealTimeLandmarkPipeline()
        try:
            assert isinstance(pipeline.detector, MediaPipeDetector)
            assert isinstance(pipeline.extractor, LandmarkExtractor)
            assert isinstance(pipeline.normalizer, LandmarkNormalizer)
            assert pipeline.inference_engine is None
        finally:
            pipeline.close()

    def dependency_injection() -> None:
        inference = MockInferenceEngine()
        pipeline, detector = make_pipeline(inference_engine=inference)
        assert pipeline.detector is detector
        assert isinstance(pipeline.extractor, MockExtractor)
        assert isinstance(pipeline.normalizer, MockNormalizer)
        assert pipeline.inference_engine is inference

    def invalid_frame_input() -> None:
        pipeline, _ = make_pipeline()
        for invalid_frame in (None, "invalid", np.array([]), np.zeros((4, 4)), np.zeros((4, 4, 1))):
            try:
                pipeline.process_frame(invalid_frame)
            except (TypeError, ValueError) as error:
                assert str(error)
            else:
                raise AssertionError("Invalid frame input was accepted")

    def feature_generation() -> None:
        pipeline, _ = make_pipeline()
        result = pipeline.process_frame(frame)
        assert result["features"].shape == (126,)
        assert np.all(np.isfinite(result["features"]))

    def no_hand_frame_behavior() -> None:
        detector = MockDetector({"success": False, "num_hands": 0, "handedness": [], "landmarks": []})
        pipeline = RealTimeLandmarkPipeline(detector, LandmarkExtractor(), LandmarkNormalizer())
        result = pipeline.process_frame(frame)
        assert result["num_hands"] == 0
        assert np.array_equal(result["features"], np.zeros(126, dtype=np.float32))

    def feature_validation_failure() -> None:
        for invalid_features in (
            np.zeros(125, dtype=np.float32),
            np.full(126, np.nan, dtype=np.float32),
            np.full(126, np.inf, dtype=np.float32),
        ):
            pipeline, _ = make_pipeline(output=invalid_features)
            try:
                pipeline.process_frame(frame)
            except (TypeError, ValueError) as error:
                assert str(error)
            else:
                raise AssertionError("Invalid normalized features were accepted")

    def inference_before_ready() -> None:
        inference = MockInferenceEngine(ready_after=32)
        pipeline, _ = make_pipeline(inference_engine=inference)
        for _ in range(31):
            assert pipeline.process_frame(frame)["prediction"] is None
        assert len(inference.frames) == 31 and inference.predict_calls == 0

    def inference_when_ready() -> None:
        inference = MockInferenceEngine(ready_after=2)
        pipeline, _ = make_pipeline(inference_engine=inference)
        assert pipeline.process_frame(frame)["prediction"] is None
        result = pipeline.process_frame(frame)
        assert result["prediction"]["predicted_class"] == "mock_sign"
        assert inference.predict_calls == 1

    def reset_behavior() -> None:
        inference = MockInferenceEngine(ready_after=1)
        pipeline, _ = make_pipeline(inference_engine=inference)
        pipeline.process_frame(frame)
        pipeline.reset()
        assert inference.reset_called and not inference.frames

    def resource_cleanup() -> None:
        pipeline, detector = make_pipeline()
        pipeline.close()
        assert detector.close_called

    def real_component_integration() -> None:
        pipeline = RealTimeLandmarkPipeline()
        try:
            result = pipeline.process_frame(frame)
            assert result["features"].shape == (126,)
            assert np.all(np.isfinite(result["features"]))
        finally:
            pipeline.close()

    run_test("Test 1: Component initialization", component_initialization)
    run_test("Test 2: Dependency injection", dependency_injection)
    run_test("Test 3: Invalid frame input", invalid_frame_input)
    run_test("Test 4: Feature generation", feature_generation)
    run_test("Test 5: No-hand frame behavior", no_hand_frame_behavior)
    run_test("Test 6: Feature validation failure", feature_validation_failure)
    run_test("Test 7: Inference integration before ready", inference_before_ready)
    run_test("Test 8: Inference integration when ready", inference_when_ready)
    run_test("Test 9: Reset behavior", reset_behavior)
    run_test("Test 10: Resource cleanup", resource_cleanup)
    run_test("Test 11: Real component integration", real_component_integration)
    print("\nAll RealTimeLandmarkPipeline tests completed successfully!")


if __name__ == "__main__":
    main()
