"""Module containing the FrameBuffer class for sequence sliding window storage."""

import logging
from collections import deque
from typing import List
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures

logger = logging.getLogger(__name__)


class FrameBuffer:
    """A sliding window buffer that stores a fixed-length sequence of HandFeatures.

    This buffer is designed to accumulate normalized landmark vectors for feeding
    into sequence-based models (like LSTM or Transformer) for sign gesture recognition.
    """

    def __init__(self, buffer_size: int = 30) -> None:
        """Initializes the FrameBuffer with a configurable maximum size.

        Args:
            buffer_size (int): The maximum length of the sequence window. Defaults to 30.

        Raises:
            ValueError: If buffer_size is less than 1.
        """
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {buffer_size}")

        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)
        logger.info("Initialized FrameBuffer with buffer_size=%d", buffer_size)

    def add(self, features: HandFeatures) -> None:
        """Adds a normalized HandFeatures object to the sliding window buffer.

        Args:
            features (HandFeatures): The normalized hand features to add.

        Raises:
            TypeError: If features is not an instance of HandFeatures.
            ValueError: If landmarks array shape/size is incorrect.
        """
        if not isinstance(features, HandFeatures):
            raise TypeError(
                f"Expected HandFeatures object, got {type(features)}"
            )

        if not isinstance(features.landmarks, np.ndarray) or features.landmarks.shape != (63,):
            raise ValueError(
                f"Expected HandFeatures landmarks to have shape (63,), got {getattr(features.landmarks, 'shape', None)}"
            )

        # Append to the sliding window. Since maxlen is set on the deque,
        # it will automatically discard the oldest element when limit is reached.
        self._buffer.append(features)

    def is_full(self) -> bool:
        """Checks if the buffer is full.

        Returns:
            bool: True if the current size equals buffer_size, False otherwise.
        """
        return len(self._buffer) == self.buffer_size

    def clear(self) -> None:
        """Clears all stored frames from the buffer."""
        self._buffer.clear()
        logger.debug("FrameBuffer cleared.")

    def get_sequence(self) -> np.ndarray:
        """Returns the current sequence of landmarks as a single NumPy array.

        Returns:
            np.ndarray: A NumPy array of shape (N, 63), where N is the current buffer size.
                If the buffer is full, the shape will be (buffer_size, 63).
                If the buffer is empty, returns an empty array of shape (0, 63).
        """
        if not self._buffer:
            return np.empty((0, 63), dtype=np.float32)

        # Extract landmark arrays from the HandFeatures in the deque
        landmarks_list = [hf.landmarks for hf in self._buffer]
        return np.vstack(landmarks_list)

    def __len__(self) -> int:
        """Returns the number of elements currently stored in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self._buffer)
