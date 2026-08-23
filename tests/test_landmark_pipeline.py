"""Tests for the SilentVoice real-time landmark pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from backend.realtime.landmark_pipeline import RealTimeLandmarkPipeline
from backend.sign_recognition.frame_features import FrameFeatures
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.normalizer import LandmarkNormalizer


class MockDetector:
    """Controlled detector for pipeline tests without MediaPipe execution."""

    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.result = result or {"success": True, "num_hands": 1, "landmarks": []}
        self.frames: list[np.ndarray] = []
        self.close_called = False

    def detect(self, frame: np.ndarray) -> dict[str, object]:
        self.frames.append(frame)
        return self.result

    def close(self) -> None:
        self.close_called = True


class MockExtractor:
    """Extractor returning a predefined FrameFeatures-compatible object."""

    def __init__(self, output: object | None = None) -> None:
        self.output = FrameFeatures() if output is None else output
        self.results: list[dict[str, object]] = []

    def extract(self, detection_result: dict[str, object]) -> object:
        self.results.append(detection_result)
        return self.output


class MockNormalizer:
    """Normalizer returning a predefined feature vector."""

    def __init__(self, output: object | None = None) -> None:
        self.output = np.arange(126, dtype=np.float32) if output is None else output
        self.inputs: list[object] = []

    def normalize(self, frame_features: object) -> object:
        self.inputs.append(frame_features)
        return self.output


class MockInferenceEngine:
    """Small sequence-aware fake that exposes the InferenceEngine public contract."""

    def __init__(self, ready_after: int = 32) -> None:
        self.ready_after = ready_after
        self.frames: list[np.ndarray] = []
        self.predict_calls = 0
        self.reset_calls = 0

    def add_landmarks(self, features: np.ndarray) -> None:
        self.frames.append(features.copy())

    def is_ready(self) -> bool:
        return len(self.frames) >= self.ready_after

    def predict(self) -> dict[str, object]:
        self.predict_calls += 1
        return {
            "predicted_index": 2,
            "predicted_class": "mock_sign",
            "confidence": 0.9,
            "probabilities": [0.05, 0.05, 0.9],
        }

    def reset(self) -> None:
        self.frames.clear()
        self.reset_calls += 1


class TestRealTimeLandmarkPipeline(unittest.TestCase):
    """Covers composition, validation, inference handoff, and cleanup."""

    def setUp(self) -> None:
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def _mock_pipeline(
        self,
        normalizer_output: object | None = None,
        inference_engine: MockInferenceEngine | None = None,
    ) -> tuple[RealTimeLandmarkPipeline, MockDetector, MockExtractor, MockNormalizer]:
        detector = MockDetector()
        extractor = MockExtractor()
        normalizer = MockNormalizer(normalizer_output)
        return (
            RealTimeLandmarkPipeline(detector, extractor, normalizer, inference_engine),
            detector,
            extractor,
            normalizer,
        )

    def test_component_initialization(self) -> None:
        pipeline = RealTimeLandmarkPipeline()
        try:
            self.assertIsInstance(pipeline.detector, MediaPipeDetector)
            self.assertIsInstance(pipeline.extractor, LandmarkExtractor)
            self.assertIsInstance(pipeline.normalizer, LandmarkNormalizer)
            self.assertIsNone(pipeline.inference_engine)
        finally:
            pipeline.close()

    def test_dependency_injection(self) -> None:
        inference_engine = MockInferenceEngine()
        pipeline, detector, extractor, normalizer = self._mock_pipeline(
            inference_engine=inference_engine
        )
        self.assertIs(pipeline.detector, detector)
        self.assertIs(pipeline.extractor, extractor)
        self.assertIs(pipeline.normalizer, normalizer)
        self.assertIs(pipeline.inference_engine, inference_engine)

    def test_invalid_frame_input(self) -> None:
        pipeline, _, _, _ = self._mock_pipeline()
        invalid_frames = (None, "not an array", np.array([]), np.zeros((32, 32)), np.zeros((32, 32, 1)))
        for invalid_frame in invalid_frames:
            with self.subTest(invalid_frame=type(invalid_frame).__name__):
                with self.assertRaises((TypeError, ValueError)) as raised:
                    pipeline.process_frame(invalid_frame)
                self.assertTrue(str(raised.exception))

    def test_feature_generation(self) -> None:
        pipeline, _, _, _ = self._mock_pipeline()
        result = pipeline.process_frame(self.frame)
        self.assertTrue(result["success"])
        self.assertEqual(result["num_hands"], 1)
        self.assertIsNone(result["prediction"])
        self.assertIsInstance(result["features"], np.ndarray)
        self.assertEqual(result["features"].shape, (126,))
        self.assertTrue(np.all(np.isfinite(result["features"])))

    def test_no_hand_frame_produces_zero_features(self) -> None:
        detector = MockDetector(
            {"success": False, "num_hands": 0, "handedness": [], "landmarks": []}
        )
        pipeline = RealTimeLandmarkPipeline(
            detector=detector,
            extractor=LandmarkExtractor(),
            normalizer=LandmarkNormalizer(),
        )
        result = pipeline.process_frame(self.frame)
        self.assertTrue(result["success"])
        self.assertEqual(result["num_hands"], 0)
        np.testing.assert_array_equal(result["features"], np.zeros(126, dtype=np.float32))

    def test_feature_validation_failure(self) -> None:
        invalid_outputs = (
            np.zeros(125, dtype=np.float32),
            np.full(126, np.nan, dtype=np.float32),
            np.full(126, np.inf, dtype=np.float32),
        )
        for invalid_output in invalid_outputs:
            with self.subTest(shape=getattr(invalid_output, "shape", None)):
                pipeline, _, _, _ = self._mock_pipeline(normalizer_output=invalid_output)
                with self.assertRaises((TypeError, ValueError)) as raised:
                    pipeline.process_frame(self.frame)
                self.assertTrue(str(raised.exception))

    def test_inference_integration_before_ready(self) -> None:
        inference_engine = MockInferenceEngine(ready_after=32)
        pipeline, _, _, _ = self._mock_pipeline(inference_engine=inference_engine)
        for _ in range(31):
            result = pipeline.process_frame(self.frame)
            self.assertIsNone(result["prediction"])
        self.assertEqual(len(inference_engine.frames), 31)
        self.assertEqual(inference_engine.predict_calls, 0)

    def test_inference_integration_when_ready(self) -> None:
        inference_engine = MockInferenceEngine(ready_after=2)
        pipeline, _, _, _ = self._mock_pipeline(inference_engine=inference_engine)
        self.assertIsNone(pipeline.process_frame(self.frame)["prediction"])
        result = pipeline.process_frame(self.frame)
        self.assertEqual(len(inference_engine.frames), 2)
        self.assertEqual(inference_engine.predict_calls, 1)
        self.assertEqual(result["prediction"]["predicted_class"], "mock_sign")

    def test_reset_delegates_to_inference_engine(self) -> None:
        inference_engine = MockInferenceEngine(ready_after=1)
        pipeline, _, _, _ = self._mock_pipeline(inference_engine=inference_engine)
        pipeline.process_frame(self.frame)
        pipeline.reset()
        self.assertEqual(inference_engine.frames, [])
        self.assertEqual(inference_engine.reset_calls, 1)

    def test_close_delegates_to_detector(self) -> None:
        pipeline, detector, _, _ = self._mock_pipeline()
        pipeline.close()
        self.assertTrue(detector.close_called)

    def test_real_component_integration(self) -> None:
        pipeline = RealTimeLandmarkPipeline()
        try:
            result = pipeline.process_frame(self.frame)
            self.assertTrue(result["success"])
            self.assertEqual(result["features"].shape, (126,))
            self.assertTrue(np.all(np.isfinite(result["features"])))
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
