"""Frame sampler module for selecting a fixed number of frames from a sequence."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class FrameSampler:
    """Selects a fixed number of evenly distributed frames from an ordered frame sequence."""

    def __init__(self, sequence_length: int = 32) -> None:
        """Initializes the FrameSampler with a target sequence length.

        Parameters
        ----------
        sequence_length : int, default 32
            The target number of frames to return. Must be a positive integer.

        Raises
        ------
        TypeError
            If sequence_length is not an integer.
        ValueError
            If sequence_length is not a positive integer.
        """
        # In Python, bool is a subclass of int, so we explicitly check it
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            logger.error("sequence_length must be an integer, got: %s", type(sequence_length))
            raise TypeError("sequence_length must be an integer")

        if sequence_length <= 0:
            logger.error("sequence_length must be positive, got: %d", sequence_length)
            raise ValueError("sequence_length must be a positive integer")

        self.sequence_length = sequence_length
        logger.debug("Initialized FrameSampler with sequence_length=%d", self.sequence_length)

    def sample(self, frame_paths: Sequence[Path]) -> list[Path] | Sequence[Path]:
        """Samples exactly sequence_length paths from the input sequence of frame paths.

        - If the sequence has more frames than sequence_length, selects frames distributed
          uniformly, preserving the first and last frame.
        - If the sequence has exactly sequence_length, returns the original sequence.
        - If the sequence has fewer frames than sequence_length, pads the sequence by
          repeating the final frame.

        Parameters
        ----------
        frame_paths : Sequence of Path
            An ordered sequence of pathlib.Path objects representing the frames.

        Returns
        -------
        list of Path or Sequence of Path
            A sequence of exactly sequence_length Paths.

        Raises
        ------
        TypeError
            If frame_paths is not a sequence.
        ValueError
            If frame_paths is empty.
        """
        if not isinstance(frame_paths, Sequence):
            logger.error("frame_paths must be a Sequence, got: %s", type(frame_paths))
            raise TypeError("frame_paths must be a Sequence (e.g. list, tuple)")

        num_frames = len(frame_paths)
        if num_frames == 0:
            logger.error("Received empty frame_paths sequence")
            raise ValueError("frame_paths sequence cannot be empty")

        # Case 2: Input contains exactly the target number
        if num_frames == self.sequence_length:
            return frame_paths

        # Case 3: Input contains fewer frames than the target
        if num_frames < self.sequence_length:
            last_frame = frame_paths[-1]
            padding_len = self.sequence_length - num_frames
            # Return list with repetitions of the final frame
            return list(frame_paths) + [last_frame] * padding_len

        # Case 1: Input contains more frames than target
        # If target sequence length is 1, return the first frame
        if self.sequence_length == 1:
            return [frame_paths[0]]

        # Uniformly distribute indices from 0 to num_frames - 1
        indices = [
            int(round(i * (num_frames - 1) / (self.sequence_length - 1)))
            for i in range(self.sequence_length)
        ]
        return [frame_paths[idx] for idx in indices]
