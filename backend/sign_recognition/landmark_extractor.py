"""Landmark extractor module for extracting and flattening hand landmarks from detection results."""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_features import FrameFeatures

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """Extracts and flattens hand landmarks from MediaPipe detector outputs.

    This class processes the structured dictionary output from MediaPipeDetector
    and converts landmark coordinates into a FrameFeatures object containing
    HandFeatures for both left and right hands.
    """

    def __init__(self) -> None:
        """Initializes the LandmarkExtractor."""
        logger.info("Initializing LandmarkExtractor.")

    def extract(self, detection_result: Dict[str, Any]) -> FrameFeatures:
        """Extracts and flattens hand landmark coordinates from detection results.

        Args:
            detection_result (Dict[str, Any]): The structured dictionary returned by
                MediaPipeDetector.detect(), containing:
                - "success" (bool): True if hands were detected.
                - "landmarks" (List[List[Tuple[float, float, float]]]): List of 21 landmarks per hand.
                - "handedness" (List[str], optional): List of handedness labels.
                - "confidences" (List[float], optional): List of detection confidence scores.

        Returns:
            FrameFeatures: A FrameFeatures instance containing HandFeatures for left and/or right hand,
                or None for either hand if not detected or invalid.
        """
        if not detection_result or not detection_result.get("success", False):
            return FrameFeatures()

        landmarks_list = detection_result.get("landmarks", [])
        if not landmarks_list:
            return FrameFeatures()

        handedness_list = detection_result.get("handedness", [])
        confidences_list = detection_result.get("confidences", [])

        left_hand = None
        right_hand = None

        for idx, hand_landmarks in enumerate(landmarks_list):
            # Validate input data structure
            if not isinstance(hand_landmarks, list):
                logger.warning(
                    "Hand index %d: Landmarks data is not a list. Got type %s.",
                    idx,
                    type(hand_landmarks)
                )
                continue

            if len(hand_landmarks) != 21:
                logger.warning(
                    "Hand index %d: Invalid number of landmarks. Expected 21, got %d.",
                    idx,
                    len(hand_landmarks)
                )
                continue

            try:
                # Convert the 21 (x, y, z) tuples into a flat NumPy array of length 63
                coords: List[float] = []
                for lm_idx, lm in enumerate(hand_landmarks):
                    if not isinstance(lm, (list, tuple)) or len(lm) != 3:
                        raise ValueError(
                            f"Landmark {lm_idx} is not a 3-element tuple/list: {lm}"
                        )
                    coords.extend(lm)

                feature_vector = np.array(coords, dtype=np.float32)

                # Get handedness and confidence, providing safe defaults if missing or invalid
                handedness = "Right"
                if idx < len(handedness_list):
                    label = handedness_list[idx]
                    if label in ("Left", "Right"):
                        handedness = label
                    else:
                        logger.warning(
                            "Hand index %d: Invalid handedness label '%s', defaulting to 'Right'.",
                            idx,
                            label
                        )
                else:
                    logger.warning(
                        "Hand index %d: Missing handedness info, defaulting to 'Right'.",
                        idx
                    )

                confidence = 1.0
                if idx < len(confidences_list):
                    val = confidences_list[idx]
                    if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                        confidence = float(val)
                    else:
                        logger.warning(
                            "Hand index %d: Invalid confidence score %s, defaulting to 1.0.",
                            idx,
                            val
                        )
                else:
                    logger.warning(
                        "Hand index %d: Missing confidence score, defaulting to 1.0.",
                        idx
                    )

                hand_features = HandFeatures(
                    landmarks=feature_vector,
                    handedness=handedness,
                    confidence=confidence
                )
                
                if handedness == "Left":
                    left_hand = hand_features
                else:
                    right_hand = hand_features

            except Exception as e:
                logger.warning(
                    "Hand index %d: Failed to extract landmarks due to malformed data: %s",
                    idx,
                    e
                )
                continue

        return FrameFeatures(left_hand=left_hand, right_hand=right_hand)

