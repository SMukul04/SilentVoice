"""Module containing the FrameBuffer class for sequence sliding window storage."""

import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """A sliding window buffer that stores a fixed-length sequence of normalized frame features.

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

    def append(self, frame: np.ndarray) -> None:
        """Appends a normalized frame vector (concatenated left and right hand landmarks)
        to the sliding window buffer.

        Args:
            frame (np.ndarray): The normalized hand features array of shape (126,).

        Raises:
            TypeError: If frame is not a NumPy array.
            ValueError: If frame shape/size is incorrect (not (126,)).
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(frame)}"
            )

        if frame.shape != (126,):
            raise ValueError(
                f"Expected frame shape to be (126,), got {frame.shape}"
            )

        # Append to the sliding window. Since maxlen is set on the deque,
        # it will automatically discard the oldest element when limit is reached.
        self._buffer.append(frame)

    def add(self, frame: np.ndarray) -> None:
        """Alias for append to preserve backwards compatibility.

        Args:
            frame (np.ndarray): The normalized hand features array of shape (126,).
        """
        self.append(frame)

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

    def size(self) -> int:
        """Returns the current number of frames stored in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self._buffer)

    def get_sequence(self) -> np.ndarray:
        """Returns the current sequence of landmarks as a single NumPy array.

        Returns:
            np.ndarray: A NumPy array of shape (N, 126), where N is the current buffer size.
                If the buffer is full, the shape will be (buffer_size, 126).
                If the buffer is empty, returns an empty array of shape (0, 126).
        """
        if not self._buffer:
            return np.empty((0, 126), dtype=np.float32)

        return np.vstack(list(self._buffer))

    def __len__(self) -> int:
        """Returns the number of elements currently stored in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.size()

