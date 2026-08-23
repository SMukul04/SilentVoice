"""Composable real-time frame-to-landmark inference pipeline."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.normalizer import LandmarkNormalizer


logger = logging.getLogger(__name__)


class DetectorProtocol(Protocol):
    """Minimal detector interface used by :class:`RealTimeLandmarkPipeline`."""

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Return the existing SilentVoice hand-detection result structure."""


class ExtractorProtocol(Protocol):
    """Minimal landmark extractor interface used by the pipeline."""

    def extract(self, detection_result: dict[str, Any]) -> Any:
        """Convert a detector result into the existing FrameFeatures representation."""


class NormalizerProtocol(Protocol):
    """Minimal landmark normalizer interface used by the pipeline."""

    def normalize(self, frame_features: Any) -> np.ndarray:
        """Return the existing left-hand-then-right-hand feature vector."""


class InferenceEngineProtocol(Protocol):
    """Minimal InferenceEngine interface required for optional prediction."""

    def add_landmarks(self, features: np.ndarray) -> None:
        """Append one normalized landmark frame."""

    def is_ready(self) -> bool:
        """Return whether a complete model sequence is available."""

    def predict(self) -> dict[str, Any]:
        """Run a prediction on the accumulated sequence."""

    def reset(self) -> None:
        """Clear the accumulated sequence."""


class RealTimeLandmarkPipeline:
    """Bridges BGR frames to SilentVoice's normalized 126-feature convention.

    The pipeline deliberately delegates hand detection, landmark extraction, and
    normalization to the existing SilentVoice components. Its feature output is a
    ``numpy.float32`` vector ordered as ``[left_hand (63), right_hand (63)]``;
    absent hands remain represented by their existing zero-filled segments.
    """

    FEATURE_DIMENSION = 126

    def __init__(
        self,
        detector: DetectorProtocol | None = None,
        extractor: ExtractorProtocol | None = None,
        normalizer: NormalizerProtocol | None = None,
        inference_engine: InferenceEngineProtocol | None = None,
    ) -> None:
        """Initialize reusable pipeline components, allowing dependency injection.

        Args:
            detector: Existing detector implementing ``detect(frame)``. Defaults to
                :class:`MediaPipeDetector`.
            extractor: Existing extractor implementing ``extract(result)``. Defaults
                to :class:`LandmarkExtractor`.
            normalizer: Existing normalizer implementing ``normalize(features)``.
                Defaults to :class:`LandmarkNormalizer`.
            inference_engine: Optional existing InferenceEngine-compatible object.

        Raises:
            TypeError: If an injected component does not expose its required API.
        """
        self.detector = detector if detector is not None else MediaPipeDetector()
        self.extractor = extractor if extractor is not None else LandmarkExtractor()
        self.normalizer = normalizer if normalizer is not None else LandmarkNormalizer()
        self.inference_engine = inference_engine

        self._validate_component(self.detector, "detector", "detect")
        self._validate_component(self.extractor, "extractor", "extract")
        self._validate_component(self.normalizer, "normalizer", "normalize")
        if self.inference_engine is not None:
            for method_name in ("add_landmarks", "is_ready", "predict", "reset"):
                self._validate_component(self.inference_engine, "inference_engine", method_name)

        logger.info(
            "Initialized RealTimeLandmarkPipeline (inference_enabled=%s)",
            self.inference_engine is not None,
        )

    @staticmethod
    def _validate_component(component: Any, component_name: str, method_name: str) -> None:
        """Ensure an injected dependency provides a callable required method."""
        if not callable(getattr(component, method_name, None)):
            raise TypeError(
                f"{component_name} must provide a callable {method_name}() method"
            )

    @staticmethod
    def validate_frame(frame: Any) -> np.ndarray:
        """Validate one non-empty BGR image frame.

        Valid frames are numeric NumPy arrays with shape ``(height, width, 3)``.
        The method intentionally does not capture frames or use OpenCV itself.
        """
        if frame is None:
            raise TypeError("Frame cannot be None")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Frame must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.size == 0:
            raise ValueError("Frame cannot be empty")
        if frame.ndim != 3:
            raise ValueError(
                f"Frame must have 3 dimensions (height, width, BGR channels), got {frame.ndim}"
            )
        if frame.shape[0] <= 0 or frame.shape[1] <= 0 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must have shape (height, width, 3), got {frame.shape}"
            )
        if frame.dtype.kind not in ("u", "i", "f"):
            raise TypeError(f"Frame must contain numeric BGR values, got dtype {frame.dtype}")
        if frame.dtype.kind == "f" and not np.all(np.isfinite(frame)):
            raise ValueError("Frame contains NaN or infinite values")
        return frame

    @classmethod
    def validate_features(cls, features: Any) -> np.ndarray:
        """Validate and copy a normalized SilentVoice landmark vector.

        Returns a float32 vector of shape ``(126,)`` compatible with the trained
        LSTM model and the existing :class:`InferenceEngine`.
        """
        if not isinstance(features, np.ndarray):
            raise TypeError(
                f"Normalized features must be a numpy.ndarray, got {type(features).__name__}"
            )
        if features.dtype.kind not in ("u", "i", "f"):
            raise TypeError(f"Normalized features must be numeric, got dtype {features.dtype}")
        if features.ndim != 1 or features.shape != (cls.FEATURE_DIMENSION,):
            raise ValueError(
                "Normalized features must have shape "
                f"({cls.FEATURE_DIMENSION},), got {features.shape}"
            )

        validated_features = features.astype(np.float32, copy=True)
        if not np.all(np.isfinite(validated_features)):
            raise ValueError("Normalized features contain NaN or infinite values")
        return validated_features

    def process_frame(self, frame: Any) -> dict[str, Any]:
        """Process one BGR frame into normalized landmarks and an optional prediction.

        No-hand frames are successful frames: the existing extractor and normalizer
        produce the expected all-zero missing-hand representation, which is added to
        the supplied inference engine when one is configured.
        """
        validated_frame = self.validate_frame(frame)
        detection_result = self.detector.detect(validated_frame)
        if not isinstance(detection_result, dict):
            raise TypeError("detector.detect() must return a dictionary result")

        frame_features = self.extractor.extract(detection_result)
        features = self.validate_features(self.normalizer.normalize(frame_features))

        num_hands = detection_result.get("num_hands", 0)
        if not isinstance(num_hands, int) or isinstance(num_hands, bool) or num_hands < 0:
            raise ValueError("Detection result num_hands must be a non-negative integer")

        prediction: dict[str, Any] | None = None
        if self.inference_engine is not None:
            self.inference_engine.add_landmarks(features)
            if self.inference_engine.is_ready():
                prediction = self.inference_engine.predict()

        return {
            "success": True,
            "num_hands": num_hands,
            "features": features,
            "prediction": prediction,
        }

    def reset(self) -> None:
        """Reset only the configured inference sequence buffer, if present."""
        if self.inference_engine is not None:
            self.inference_engine.reset()
            logger.debug("Reset inference engine sequence through landmark pipeline")

    def close(self) -> None:
        """Release detector resources when the detector provides ``close()``."""
        close_method = getattr(self.detector, "close", None)
        if callable(close_method):
            close_method()
            logger.info("Closed real-time landmark detector resources")

    def __enter__(self) -> "RealTimeLandmarkPipeline":
        """Support context-managed resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release detector resources on context-manager exit."""
        self.close()
