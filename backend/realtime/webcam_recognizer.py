"""OpenCV webcam orchestration for real-time SilentVoice recognition."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.inference.inference_engine import InferenceEngine
from backend.realtime.landmark_pipeline import RealTimeLandmarkPipeline
from backend.realtime.prediction_stabilizer import PredictionStabilizer


logger = logging.getLogger(__name__)


class WebcamRecognizer:
    """Owns webcam presentation while reusing SilentVoice inference components."""

    def __init__(
        self,
        camera_index: int = 0,
        confidence_threshold: float = 0.0,
        window_name: str = "SilentVoice Recognition",
        mirror: bool = True,
        show_landmarks: bool = True,
        pipeline: RealTimeLandmarkPipeline | Any | None = None,
        inference_engine: InferenceEngine | Any | None = None,
        stabilizer: PredictionStabilizer | Any | None = None,
        cv2_module: Any = cv2,
    ) -> None:
        """Initialize reusable recognition dependencies without opening a camera.

        Dependency injection is provided for tests; production defaults load the
        existing model once and use the existing landmark pipeline unchanged.
        The pipeline has no inference engine attached because this class must keep
        no-hand frames out of the model sequence.
        """
        if not isinstance(camera_index, int) or isinstance(camera_index, bool) or camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer")
        if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool):
            raise TypeError("confidence_threshold must be a number")
        if not 0.0 <= float(confidence_threshold):
            raise ValueError("confidence_threshold cannot be negative")
        if not isinstance(window_name, str) or not window_name.strip():
            raise ValueError("window_name must be a non-empty string")
        if not isinstance(mirror, bool) or not isinstance(show_landmarks, bool):
            raise TypeError("mirror and show_landmarks must be booleans")

        self.camera_index = camera_index
        self.confidence_threshold = float(confidence_threshold)
        self.window_name = window_name
        self.mirror = mirror
        self.show_landmarks = show_landmarks
        self._cv2 = cv2_module
        self.camera: Any | None = None

        if inference_engine is None:
            inference_engine = InferenceEngine(confidence_threshold=self.confidence_threshold)
            inference_engine.load()
        self.inference_engine = inference_engine
        self.pipeline = pipeline if pipeline is not None else RealTimeLandmarkPipeline()
        self.stabilizer = stabilizer if stabilizer is not None else PredictionStabilizer(
            window_size=5,
            confidence_threshold=self.confidence_threshold,
            min_consistent_predictions=3,
        )

        for component, method in ((self.pipeline, "process_frame"), (self.inference_engine, "add_landmarks"),
                                  (self.inference_engine, "is_ready"), (self.inference_engine, "predict"),
                                  (self.inference_engine, "reset")):
            if not callable(getattr(component, method, None)):
                raise TypeError(f"Injected component must provide callable {method}()")
        for method in ("add_prediction", "reset", "is_stable"):
            if not callable(getattr(self.stabilizer, method, None)):
                raise TypeError(f"Injected stabilizer must provide callable {method}()")

        self.last_prediction: dict[str, Any] | None = None
        self.last_confidence: float | None = None
        logger.info("Initialized WebcamRecognizer for camera index %d", self.camera_index)

    def open_camera(self) -> None:
        """Open the configured OpenCV camera, failing clearly when unavailable."""
        if self.camera is not None and self.camera.isOpened():
            return
        camera = self._cv2.VideoCapture(self.camera_index)
        if camera is None or not camera.isOpened():
            if camera is not None:
                camera.release()
            raise RuntimeError(f"Unable to open webcam at camera index {self.camera_index}")
        self.camera = camera
        logger.info("Opened webcam at camera index %d", self.camera_index)

    def close_camera(self) -> None:
        """Release the webcam and destroy the recognizer's OpenCV windows."""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        finally:
            try:
                self._cv2.destroyAllWindows()
            finally:
                close_method = getattr(self.pipeline, "close", None)
                if callable(close_method):
                    close_method()

    def reset(self) -> None:
        """Clear model sequence state and the persistent display prediction."""
        self.inference_engine.reset()
        self.stabilizer.reset()
        self.last_prediction = None
        self.last_confidence = None

    @property
    def sequence_progress(self) -> tuple[int, int]:
        """Return current and required frame counts for the display overlay."""
        current = getattr(self.inference_engine, "sequence_count", 0)
        required = getattr(self.inference_engine, "sequence_length", 32)
        return int(current), int(required)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Process one captured frame and update persistent recognition state."""
        display_frame = self._cv2.flip(frame, 1) if self.mirror else frame
        pipeline_result = self.pipeline.process_frame(display_frame)
        if not isinstance(pipeline_result, dict):
            raise TypeError("pipeline.process_frame() must return a dictionary result")

        num_hands = pipeline_result.get("num_hands", 0)
        if not isinstance(num_hands, int) or isinstance(num_hands, bool) or num_hands < 0:
            raise ValueError("Pipeline result num_hands must be a non-negative integer")

        prediction: dict[str, Any] | None = None
        if num_hands > 0:
            features = pipeline_result.get("features")
            self.inference_engine.add_landmarks(features)
            if self.inference_engine.is_ready():
                raw_prediction = self.inference_engine.predict()
                prediction = self.stabilizer.add_prediction(raw_prediction)
                self.last_prediction = prediction
                self.last_confidence = float(prediction["confidence"])

        return display_frame, {
            "num_hands": num_hands,
            "prediction": prediction,
            "last_prediction": self.last_prediction,
            "is_stable": self.stabilizer.is_stable(),
            "stabilizer_history_size": getattr(self.stabilizer, "history_size", 0),
            "sequence_progress": self.sequence_progress,
        }

    def draw_overlay(self, frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        """Draw hand status, sequence progress, and the persistent prediction."""
        output = frame
        if self.show_landmarks:
            draw_method = getattr(getattr(self.pipeline, "detector", None), "draw", None)
            if callable(draw_method):
                output = draw_method(output)

        num_hands = state["num_hands"]
        current, required = state["sequence_progress"]
        if num_hands == 0:
            status = "No hands detected"
        elif self.last_prediction is None:
            status = "Sign: Collecting frames"
        elif not state["is_stable"]:
            status = "Sign: Stabilizing..."
        else:
            status = f"Sign: {self.last_prediction['predicted_class']}"

        self._cv2.putText(output, status, (16, 32), self._cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._cv2.putText(output, f"Frames: {current}/{required}", (16, 62), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if self.last_prediction is not None and self.last_confidence is not None:
            self._cv2.putText(output, f"Confidence: {self.last_confidence * 100:.2f}%", (16, 92), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return output

    def run(self) -> None:
        """Run the webcam loop until Q or ESC is pressed, always cleaning up."""
        try:
            self.open_camera()
            while True:
                success, frame = self.camera.read()
                if not success or frame is None:
                    logger.warning("Unable to read webcam frame; stopping recognition")
                    break
                display_frame, state = self.process_frame(frame)
                self._cv2.imshow(self.window_name, self.draw_overlay(display_frame, state))
                key = self._cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    self.reset()
        finally:
            self.close_camera()

    def __enter__(self) -> "WebcamRecognizer":
        """Open the camera for context-managed use."""
        self.open_camera()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Ensure camera and detector resources are released."""
        self.close_camera()
