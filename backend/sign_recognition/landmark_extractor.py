"""Landmark extractor module for extracting and flattening hand landmarks from detection results."""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """Extracts and flattens hand landmarks from MediaPipe detector outputs.

    This class processes the structured dictionary output from MediaPipeDetector
    and converts landmark coordinates into flattened NumPy feature vectors of length 63.
    """

    def __init__(self) -> None:
        """Initializes the LandmarkExtractor."""
        logger.info("Initializing LandmarkExtractor.")

    def extract(self, detection_result: Dict[str, Any]) -> List[np.ndarray]:
        """Extracts and flattens hand landmark coordinates from detection results.

        Args:
            detection_result (Dict[str, Any]): The structured dictionary returned by
                MediaPipeDetector.detect(), containing:
                - "success" (bool): True if hands were detected.
                - "landmarks" (List[List[Tuple[float, float, float]]]): List of 21 landmarks per hand.

        Returns:
            List[np.ndarray]: A list of 1D NumPy arrays of shape (63,), one per valid detected hand.
                If no hands are detected or data is invalid, returns an empty list.
        """
        if not detection_result or not detection_result.get("success", False):
            return []

        landmarks_list = detection_result.get("landmarks", [])
        if not landmarks_list:
            return []

        extracted_vectors: List[np.ndarray] = []

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
                # We expect each element to be a sequence of 3 floats (x, y, z)
                coords: List[float] = []
                for lm_idx, lm in enumerate(hand_landmarks):
                    if not isinstance(lm, (list, tuple)) or len(lm) != 3:
                        raise ValueError(
                            f"Landmark {lm_idx} is not a 3-element tuple/list: {lm}"
                        )
                    coords.extend(lm)

                feature_vector = np.array(coords, dtype=np.float32)
                extracted_vectors.append(feature_vector)

            except Exception as e:
                logger.warning(
                    "Hand index %d: Failed to extract landmarks due to malformed data: %s",
                    idx,
                    e
                )
                continue

        return extracted_vectors
