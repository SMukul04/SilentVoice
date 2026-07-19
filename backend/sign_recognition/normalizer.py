"""Module for normalizing hand landmark coordinates using translation and scaling."""

import logging
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_features import FrameFeatures

logger = logging.getLogger(__name__)


class LandmarkNormalizer:
    """Normalizes HandFeatures landmarks to be translation and scale invariant.

    This normalizer translates landmarks so that the wrist (landmark 0) is at (0, 0, 0),
    and scales the coordinates by a reference distance (distance between wrist and middle finger MCP).
    """

    def __init__(self) -> None:
        """Initializes the LandmarkNormalizer."""
        logger.info("Initializing LandmarkNormalizer.")

    def _normalize_hand(self, hand_features: HandFeatures) -> np.ndarray:
        """Helper method to normalize a single hand's landmarks.

        Args:
            hand_features (HandFeatures): The original hand features object.

        Returns:
            np.ndarray: A normalized flat float32 array of shape (63,).

        Raises:
            TypeError: If hand_features is not an instance of HandFeatures.
            ValueError: If landmarks array shape/size is incorrect.
        """
        if not isinstance(hand_features, HandFeatures):
            raise TypeError(
                f"Expected HandFeatures object, got {type(hand_features)}"
            )

        landmarks = hand_features.landmarks
        if landmarks.shape != (63,):
            raise ValueError(
                f"Expected landmarks array of shape (63,), got {landmarks.shape}"
            )

        # Reshape to (21, 3) to make translation and scaling math easier
        coords = landmarks.reshape(21, 3)

        # 1. Translation: Move landmark 0 (wrist) to origin (0, 0, 0)
        wrist = coords[0]
        translated = coords - wrist

        # 2. Scaling: Calculate reference distance from wrist (landmark 0) to middle MCP (landmark 9)
        # In translated coordinates, wrist is at translated[0], which is (0, 0, 0)
        # So reference distance is the Euclidean norm of translated[9] (landmark 9 coords)
        ref_vector = translated[9]
        ref_distance = float(np.linalg.norm(ref_vector))

        if ref_distance > 1e-6:
            normalized = translated / ref_distance
        else:
            logger.warning(
                "Reference distance (wrist to middle MCP) is too small (%f). Skipping scaling normalization.",
                ref_distance
            )
            normalized = translated

        # Flatten back to shape (63,) and convert to float32
        return normalized.flatten().astype(np.float32)

    def normalize(self, frame_features: FrameFeatures) -> np.ndarray:
        """Normalizes both left and right hands from frame features and concatenates them.

        Args:
            frame_features (FrameFeatures): The frame features object containing left
                and right hands.

        Returns:
            np.ndarray: A concatenated NumPy array of shape (126,) containing
                [left_hand_normalized, right_hand_normalized]. Missing hands are
                represented by exactly 63 zeros.

        Raises:
            TypeError: If frame_features is not an instance of FrameFeatures.
        """
        if not isinstance(frame_features, FrameFeatures):
            raise TypeError(
                f"Expected FrameFeatures object, got {type(frame_features)}"
            )

        # Process Left Hand
        if frame_features.left_hand is not None:
            left_norm = self._normalize_hand(frame_features.left_hand)
        else:
            left_norm = np.zeros(63, dtype=np.float32)

        # Process Right Hand
        if frame_features.right_hand is not None:
            right_norm = self._normalize_hand(frame_features.right_hand)
        else:
            right_norm = np.zeros(63, dtype=np.float32)

        # Concatenate and return
        return np.concatenate([left_norm, right_norm]).astype(np.float32)
