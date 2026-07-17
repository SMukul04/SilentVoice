"""Module for normalizing hand landmark coordinates using translation and scaling."""

import logging
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures

logger = logging.getLogger(__name__)


class LandmarkNormalizer:
    """Normalizes HandFeatures landmarks to be translation and scale invariant.

    This normalizer translates landmarks so that the wrist (landmark 0) is at (0, 0, 0),
    and scales the coordinates by a reference distance (distance between wrist and middle finger MCP).
    """

    def __init__(self) -> None:
        """Initializes the LandmarkNormalizer."""
        logger.info("Initializing LandmarkNormalizer.")

    def normalize(self, hand_features: HandFeatures) -> HandFeatures:
        """Normalizes landmarks using translation and scaling.

        Args:
            hand_features (HandFeatures): The original hand features object.

        Returns:
            HandFeatures: A new HandFeatures object containing the normalized landmarks.

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
        normalized_flat = normalized.flatten().astype(np.float32)

        # Return a brand new HandFeatures object to keep original instance immutable
        return HandFeatures(
            landmarks=normalized_flat,
            handedness=hand_features.handedness,
            confidence=hand_features.confidence
        )
