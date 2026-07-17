"""Module defining the HandFeatures dataclass to encapsulate hand features and metadata."""

from dataclasses import dataclass
import numpy as np


@dataclass
class HandFeatures:
    """Encapsulates hand landmark features and associated metadata.

    Attributes:
        landmarks (np.ndarray): Flattened landmark feature vector of shape (63,).
        handedness (str): Handedness label, either "Left" or "Right".
        confidence (float): The detection confidence score from MediaPipe.
    """
    landmarks: np.ndarray
    handedness: str
    confidence: float

    def __post_init__(self) -> None:
        """Validates the input parameters after initialization.

        Raises:
            TypeError: If landmarks is not a NumPy array or handedness/confidence has wrong type.
            ValueError: If landmarks array shape/length is not exactly 63 or handedness is invalid.
        """
        # Validate landmarks type and structure
        if not isinstance(self.landmarks, np.ndarray):
            raise TypeError(
                f"landmarks must be a numpy.ndarray, got {type(self.landmarks)}"
            )

        if self.landmarks.ndim != 1 or self.landmarks.shape[0] != 63:
            raise ValueError(
                f"landmarks must be a 1D NumPy array of shape (63,), got shape {self.landmarks.shape}"
            )

        # Validate handedness
        if not isinstance(self.handedness, str):
            raise TypeError(
                f"handedness must be a string, got {type(self.handedness)}"
            )

        if self.handedness not in ("Left", "Right"):
            raise ValueError(
                f"handedness must be either 'Left' or 'Right', got '{self.handedness}'"
            )

        # Validate confidence
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a float, got {type(self.confidence)}"
            )

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
