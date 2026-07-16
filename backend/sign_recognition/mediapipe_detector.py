"""MediaPipe Hands detector module optimized for real-time inference."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

from backend.config import (
    HAND_STATIC_IMAGE_MODE,
    HAND_MAX_NUM_HANDS,
    HAND_MODEL_COMPLEXITY,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    SWAP_HANDEDNESS,
)

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """Detects and tracks hands in video frames using MediaPipe Hands.

    This class encapsulates MediaPipe Hands initialization, processing, drawing,
    and resource cleanup. It is designed to be initialized once and reused
    for real-time webcam inference.
    """

    def __init__(self) -> None:
        """Initializes the MediaPipe Hands detector using configuration values."""
        logger.info("Initializing MediaPipe Hands Detector...")
        try:
            self.mp_hands = mp_hands
            self.mp_drawing = mp_drawing
            self.mp_drawing_styles = mp_drawing_styles

            self.hands = self.mp_hands.Hands(
                static_image_mode=HAND_STATIC_IMAGE_MODE,
                max_num_hands=HAND_MAX_NUM_HANDS,
                model_complexity=HAND_MODEL_COMPLEXITY,
                min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
            )
            # Store latest raw results from MediaPipe for use in drawing or other extraction steps
            self._latest_raw_results: Any = None
            logger.info("MediaPipe Hands Detector successfully initialized.")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe Hands: %s", e)
            raise RuntimeError(f"MediaPipe initialization failed: {e}") from e

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Processes a single BGR frame and returns structured hand detection details.

        Args:
            frame (np.ndarray): The raw BGR input image frame from a camera or file.

        Returns:
            Dict[str, Any]: A structured dictionary containing:
                - "success" (bool): True if at least one hand is detected.
                - "num_hands" (int): The number of detected hands (0, 1, or 2).
                - "handedness" (List[str]): List of handedness labels ('Left' or 'Right').
                  Note: MediaPipe's handedness is relative to the person in the image.
                - "landmarks" (List[List[Tuple[float, float, float]]]): List containing 21 (x, y, z)
                  tuples of landmarks for each detected hand.
        """
        if frame is None:
            logger.warning("Received empty frame for hand detection.")
            return {
                "success": False,
                "num_hands": 0,
                "handedness": [],
                "landmarks": [],
            }

        try:
            # MediaPipe expects RGB images, so convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._latest_raw_results = self.hands.process(rgb_frame)

            success = False
            num_hands = 0
            handedness_list: List[str] = []
            landmarks_list: List[List[Tuple[float, float, float]]] = []

            if self._latest_raw_results.multi_hand_landmarks:
                num_hands = len(self._latest_raw_results.multi_hand_landmarks)
                success = num_hands > 0

                # Extract handedness labels
                if self._latest_raw_results.multi_handedness:
                    for idx, hand_classification in enumerate(self._latest_raw_results.multi_handedness):
                        # classification is a list of category objects; get the label (e.g. Left/Right)
                        label = hand_classification.classification[0].label
                        
                        # Swap handedness if configured.
                        # MediaPipe Hands assumes a mirrored (selfie) camera feed, which reports 
                        # handedness backwards (Left as Right, Right as Left) on non-mirrored webcams.
                        if SWAP_HANDEDNESS:
                            if label == "Left":
                                label = "Right"
                            elif label == "Right":
                                label = "Left"
                                
                        handedness_list.append(label)

                # Extract landmark coordinates
                for hand_landmarks in self._latest_raw_results.multi_hand_landmarks:
                    current_hand_landmarks: List[Tuple[float, float, float]] = []
                    for lm in hand_landmarks.landmark:
                        current_hand_landmarks.append((lm.x, lm.y, lm.z))
                    landmarks_list.append(current_hand_landmarks)

            return {
                "success": success,
                "num_hands": num_hands,
                "handedness": handedness_list,
                "landmarks": landmarks_list,
            }

        except Exception as e:
            logger.error("Error during hand detection processing: %s", e)
            return {
                "success": False,
                "num_hands": 0,
                "handedness": [],
                "landmarks": [],
            }

    def draw(self, frame: np.ndarray, results: Optional[Any] = None) -> np.ndarray:
        """Draws detected hand landmarks and connections on the given frame.

        Args:
            frame (np.ndarray): The raw BGR frame on which to draw.
            results (Optional[Any]): The raw MediaPipe Hands results object. If None,
                uses the stored results from the most recent detect() call.

        Returns:
            np.ndarray: A copy of the input frame with drawn hand landmarks and connections.
        """
        annotated_frame = frame.copy()

        # Determine which raw MediaPipe results to draw
        raw_results = results if results is not None else self._latest_raw_results

        if raw_results is not None and hasattr(raw_results, "multi_hand_landmarks"):
            landmarks = raw_results.multi_hand_landmarks
            if landmarks:
                for hand_landmarks in landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

        return annotated_frame

    def close(self) -> None:
        """Closes the MediaPipe Hands object and releases its resources."""
        if hasattr(self, "hands") and self.hands is not None:
            logger.info("Closing MediaPipe Hands Detector...")
            try:
                self.hands.close()
            except Exception as e:
                logger.error("Error closing MediaPipe Hands instance: %s", e)

    def __enter__(self) -> "MediaPipeDetector":
        """Enables context manager usage."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Enables context manager cleanup."""
        self.close()
