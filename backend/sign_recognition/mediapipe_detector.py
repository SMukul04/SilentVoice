"""MediaPipe hand detector optimized for real-time inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from backend.config import (
    HAND_MAX_NUM_HANDS,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    SWAP_HANDEDNESS,
)

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """Detects hands using the MediaPipe Tasks HandLandmarker API.

    The public output format remains compatible with the existing
    SilentVoice landmark extraction pipeline.
    """

    def __init__(
        self,
        model_path: str | Path = "models/mediapipe/hand_landmarker.task",
    ) -> None:
        """Initialize the MediaPipe Hand Landmarker once for reuse."""

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found: {self.model_path}"
            )

        logger.info("Initializing MediaPipe Hand Landmarker...")

        try:
            options = vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(
                    model_asset_path=str(self.model_path)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=HAND_MAX_NUM_HANDS,
                min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
            )

            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                options
            )

            self._latest_result: Optional[Any] = None
            self._timestamp_ms = 0

            logger.info(
                "MediaPipe Hand Landmarker initialized successfully."
            )

        except Exception as e:
            logger.exception("Failed to initialize MediaPipe Hand Landmarker.")
            raise RuntimeError(
                f"MediaPipe initialization failed: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process one BGR frame and return hand landmark information.

        Returns a dictionary compatible with the existing SilentVoice
        LandmarkExtractor:

        {
            "success": bool,
            "num_hands": int,
            "handedness": list[str],
            "confidences": list[float],
            "landmarks": list[list[tuple[float, float, float]]],
        }
        """

        if frame is None:
            logger.warning("Received None frame for hand detection.")
            return self._empty_result()

        if not isinstance(frame, np.ndarray):
            logger.warning(
                "Invalid frame type received: %s",
                type(frame).__name__,
            )
            return self._empty_result()

        if frame.size == 0:
            logger.warning("Received empty frame for hand detection.")
            return self._empty_result()

        try:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            self._timestamp_ms += 1

            result = self.hand_landmarker.detect_for_video(
                mp_image,
                self._timestamp_ms,
            )

            self._latest_result = result

            handedness_list: List[str] = []
            confidences_list: List[float] = []
            landmarks_list: List[
                List[Tuple[float, float, float]]
            ] = []

            for index, hand_landmarks in enumerate(result.hand_landmarks):

                current_hand_landmarks: List[
                    Tuple[float, float, float]
                ] = []

                for landmark in hand_landmarks:
                    current_hand_landmarks.append(
                        (
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z),
                        )
                    )

                landmarks_list.append(current_hand_landmarks)

                label = ""
                confidence = 0.0

                if index < len(result.handedness):
                    categories = result.handedness[index]

                    if categories:
                        category = categories[0]

                        label = str(category.category_name)
                        confidence = float(category.score)

                if SWAP_HANDEDNESS:
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"

                handedness_list.append(label)
                confidences_list.append(confidence)

            num_hands = len(landmarks_list)

            return {
                "success": num_hands > 0,
                "num_hands": num_hands,
                "handedness": handedness_list,
                "confidences": confidences_list,
                "landmarks": landmarks_list,
            }

        except Exception as e:
            logger.exception(
                "Error during MediaPipe hand detection: %s",
                e,
            )
            return self._empty_result()

    def draw(
        self,
        frame: np.ndarray,
        results: Optional[Any] = None,
    ) -> np.ndarray:
        """Draw hand landmarks on a copy of the frame."""

        if frame is None:
            raise ValueError("frame cannot be None")

        annotated_frame = frame.copy()

        result = (
            results
            if results is not None
            else self._latest_result
        )

        if result is None:
            return annotated_frame

        try:
            height, width = annotated_frame.shape[:2]

            for hand_landmarks in result.hand_landmarks:

                points = []

                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                    cv2.circle(
                        annotated_frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1,
                    )

                for point_a, point_b in self._hand_connections():
                    if (
                        point_a < len(points)
                        and point_b < len(points)
                    ):
                        cv2.line(
                            annotated_frame,
                            points[point_a],
                            points[point_b],
                            (0, 255, 0),
                            2,
                        )

            return annotated_frame

        except Exception as e:
            logger.exception(
                "Failed to draw hand landmarks: %s",
                e,
            )
            return annotated_frame

    def close(self) -> None:
        """Release MediaPipe resources."""

        if (
            hasattr(self, "hand_landmarker")
            and self.hand_landmarker is not None
        ):
            try:
                self.hand_landmarker.close()
                logger.info(
                    "MediaPipe Hand Landmarker closed."
                )
            except Exception as e:
                logger.warning(
                    "Error while closing MediaPipe Hand Landmarker: %s",
                    e,
                )

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return the standard empty detection result."""

        return {
            "success": False,
            "num_hands": 0,
            "handedness": [],
            "confidences": [],
            "landmarks": [],
        }

    @staticmethod
    def _hand_connections() -> List[Tuple[int, int]]:
        """Return the standard MediaPipe hand landmark connections."""

        return [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]