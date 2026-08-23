"""WebcamRecognizer tests using fakes only; no physical webcam is required."""

from __future__ import annotations

import unittest

import numpy as np

from backend.realtime.webcam_recognizer import WebcamRecognizer
from backend.realtime.prediction_stabilizer import PredictionStabilizer


class FakeEngine:
    def __init__(self, ready_after: int = 32) -> None:
        self.ready_after = ready_after
        self.sequence_length = 32
        self.frames: list[np.ndarray] = []
        self.predict_calls = 0
        self.reset_calls = 0

    @property
    def sequence_count(self) -> int:
        return len(self.frames)

    def add_landmarks(self, features: np.ndarray) -> None:
        self.frames.append(features.copy())

    def is_ready(self) -> bool:
        return len(self.frames) >= self.ready_after

    def predict(self) -> dict[str, object]:
        self.predict_calls += 1
        return {"predicted_index": 1, "predicted_class": "alive", "confidence": 0.9, "probabilities": [0.1, 0.9]}

    def reset(self) -> None:
        self.frames.clear()
        self.reset_calls += 1


class FakePipeline:
    def __init__(self, num_hands: int = 1) -> None:
        self.num_hands = num_hands
        self.frames: list[np.ndarray] = []
        self.close_calls = 0

    def process_frame(self, frame: np.ndarray) -> dict[str, object]:
        self.frames.append(frame.copy())
        return {"success": True, "num_hands": self.num_hands, "features": np.ones(126, dtype=np.float32), "prediction": None}

    def close(self) -> None:
        self.close_calls += 1


class FakeStabilizer:
    def __init__(self) -> None:
        self.received: list[dict[str, object]] = []
        self.reset_calls = 0

    @property
    def history_size(self) -> int:
        return len(self.received)

    def add_prediction(self, prediction: dict[str, object]) -> dict[str, object]:
        self.received.append(prediction)
        return {**prediction, "predicted_class": "stable_alive"}

    def is_stable(self) -> bool:
        return bool(self.received)

    def reset(self) -> None:
        self.received.clear(); self.reset_calls += 1


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.release_calls += 1


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture: FakeCapture | None = None) -> None:
        self.capture = capture or FakeCapture()
        self.flip_calls = 0
        self.destroy_calls = 0

    def VideoCapture(self, index: int) -> FakeCapture:
        return self.capture

    def flip(self, frame: np.ndarray, code: int) -> np.ndarray:
        self.flip_calls += 1
        return np.flip(frame, axis=1)

    def destroyAllWindows(self) -> None:
        self.destroy_calls += 1

    def putText(self, frame: np.ndarray, *args: object) -> np.ndarray:
        return frame


class TestWebcamRecognizer(unittest.TestCase):
    def make_recognizer(self, *, num_hands: int = 1, ready_after: int = 32, mirror: bool = True, stabilizer=None) -> tuple[WebcamRecognizer, FakePipeline, FakeEngine, FakeCV2]:
        pipeline, engine, cv = FakePipeline(num_hands), FakeEngine(ready_after), FakeCV2()
        return WebcamRecognizer(pipeline=pipeline, inference_engine=engine, cv2_module=cv, mirror=mirror, stabilizer=stabilizer), pipeline, engine, cv

    def test_component_initialization(self) -> None:
        recognizer, pipeline, engine, _ = self.make_recognizer()
        self.assertIs(recognizer.pipeline, pipeline)
        self.assertIs(recognizer.inference_engine, engine)
        self.assertIsInstance(recognizer.stabilizer, PredictionStabilizer)
        self.assertTrue(recognizer.mirror)

    def test_custom_stabilizer_dependency_injection(self) -> None:
        stabilizer = FakeStabilizer()
        recognizer, _, _, _ = self.make_recognizer(stabilizer=stabilizer)
        self.assertIs(recognizer.stabilizer, stabilizer)

    def test_invalid_camera_handling(self) -> None:
        with self.assertRaises(ValueError):
            WebcamRecognizer(camera_index=-1, pipeline=FakePipeline(), inference_engine=FakeEngine(), cv2_module=FakeCV2())
        recognizer, _, _, _ = self.make_recognizer()
        recognizer._cv2 = FakeCV2(FakeCapture(opened=False))
        with self.assertRaisesRegex(RuntimeError, "Unable to open webcam"):
            recognizer.open_camera()

    def test_frame_processing_with_valid_pipeline(self) -> None:
        recognizer, _, engine, _ = self.make_recognizer()
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(len(engine.frames), 1)
        self.assertIsNone(state["prediction"])

    def test_no_hand_behavior_preserves_sequence(self) -> None:
        recognizer, _, engine, _ = self.make_recognizer(num_hands=0)
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(len(engine.frames), 0)
        self.assertEqual(state["num_hands"], 0)

    def test_sequence_progress(self) -> None:
        recognizer, _, _, _ = self.make_recognizer()
        recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(recognizer.sequence_progress, (1, 32))

    def test_prediction_when_ready(self) -> None:
        recognizer, _, engine, _ = self.make_recognizer(ready_after=2)
        recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(engine.predict_calls, 1)
        self.assertEqual(state["prediction"]["predicted_class"], "unknown")
        self.assertFalse(state["is_stable"])
        recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(state["prediction"]["predicted_class"], "alive")
        self.assertTrue(state["is_stable"])

    def test_raw_prediction_passes_through_stabilizer(self) -> None:
        stabilizer = FakeStabilizer()
        recognizer, _, engine, _ = self.make_recognizer(ready_after=1, stabilizer=stabilizer)
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(engine.predict_calls, 1)
        self.assertEqual(stabilizer.received[0]["predicted_class"], "alive")
        self.assertEqual(state["last_prediction"]["predicted_class"], "stable_alive")

    def test_prediction_state_persistence(self) -> None:
        recognizer, pipeline, _, _ = self.make_recognizer(ready_after=1)
        recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        pipeline.num_hands = 0
        _, state = recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(state["last_prediction"]["predicted_class"], "unknown")

    def test_reset_behavior(self) -> None:
        stabilizer = FakeStabilizer()
        recognizer, _, engine, _ = self.make_recognizer(ready_after=1, stabilizer=stabilizer)
        recognizer.process_frame(np.zeros((4, 5, 3), dtype=np.uint8))
        recognizer.reset()
        self.assertEqual(engine.reset_calls, 1)
        self.assertEqual(stabilizer.reset_calls, 1)
        self.assertIsNone(recognizer.last_prediction)

    def test_mirror_frame_behavior(self) -> None:
        recognizer, pipeline, _, cv = self.make_recognizer(mirror=True)
        frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        recognizer.process_frame(frame)
        np.testing.assert_array_equal(pipeline.frames[0], np.flip(frame, axis=1))
        self.assertEqual(cv.flip_calls, 1)

    def test_resource_cleanup(self) -> None:
        recognizer, pipeline, _, cv = self.make_recognizer()
        recognizer.open_camera()
        capture = recognizer.camera
        recognizer.close_camera()
        self.assertEqual(capture.release_calls, 1)
        self.assertEqual(cv.destroy_calls, 1)
        self.assertEqual(pipeline.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
