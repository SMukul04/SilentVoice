"""Landmark dataset builder module for converting video frame paths into sequence features."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
import cv2
import numpy as np

from backend.dataset.frame_sampler import FrameSampler
from backend.dataset.indexer import DatasetSample
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.normalizer import LandmarkNormalizer

# Set up logging
logger = logging.getLogger(__name__)


class LandmarkDatasetBuilder:
    """Processes video samples to build normalized landmark feature sequences."""

    def __init__(
        self,
        frame_sampler: FrameSampler,
        detector: MediaPipeDetector,
        extractor: LandmarkExtractor,
        normalizer: LandmarkNormalizer,
    ) -> None:
        """Initializes the LandmarkDatasetBuilder with its dependent pipeline stages.

        Parameters
        ----------
        frame_sampler : FrameSampler
            The sampler used to select a fixed number of frames.
        detector : MediaPipeDetector
            The MediaPipe hand detector.
        extractor : LandmarkExtractor
            The landmark extractor which formats raw coordinate outputs.
        normalizer : LandmarkNormalizer
            The normalizer which makes landmarks translation and scale invariant.
        """
        self.frame_sampler = frame_sampler
        self.detector = detector
        self.extractor = extractor
        self.normalizer = normalizer
        logger.info("LandmarkDatasetBuilder successfully initialized.")

    def build_sample(self, sample: DatasetSample) -> np.ndarray:
        """Processes a single dataset sample into a sequence of normalized landmarks.

        Parameters
        ----------
        sample : DatasetSample
            The sample object containing the video directory and frame paths.

        Returns
        -------
        np.ndarray
            A 2D array of shape (sequence_length, 126) with float32 landmarks.

        Raises
        ------
        TypeError
            If the sample input is not a DatasetSample or is missing required fields.
        FileNotFoundError
            If any frame path does not exist on disk or fails to load.
        ValueError
            If the processed sequence has an unexpected length or shape.
        """
        # Validate sample input type
        if not hasattr(sample, "frame_paths") or not hasattr(sample, "sample_id"):
            logger.error("Invalid sample object passed to build_sample")
            raise TypeError("sample must have 'frame_paths' and 'sample_id' attributes")

        # Get the target sequence length from the frame sampler
        target_len = self.frame_sampler.sequence_length

        # Obtain exactly target_len frames
        sampled_frames = self.frame_sampler.sample(sample.frame_paths)

        if len(sampled_frames) != target_len:
            logger.error(
                "Sampled frame sequence length (%d) does not match target (%d)",
                len(sampled_frames),
                target_len,
            )
            raise ValueError(
                f"Sampled frame sequence length ({len(sampled_frames)}) does not match "
                f"sampler sequence length ({target_len})"
            )

        frame_features_list: list[np.ndarray] = []

        for frame_path in sampled_frames:
            # Validate frame path exists
            if not frame_path.exists():
                logger.error("Frame path does not exist: %s", frame_path)
                raise FileNotFoundError(f"Frame path does not exist: {frame_path}")

            # Load frame image
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.error("Failed to load frame image: %s", frame_path)
                raise FileNotFoundError(f"Failed to load frame image: {frame_path}")

            # Run MediaPipe hand detection
            detection_result = self.detector.detect(frame)

            # Extract hand landmarks
            frame_features = self.extractor.extract(detection_result)

            # Normalize and flatten landmarks
            normalized_vec = self.normalizer.normalize(frame_features)

            # Verify the frame feature vector has the correct dimension (126)
            if normalized_vec.shape != (126,):
                logger.error("Frame features shape is incorrect: %s", normalized_vec.shape)
                raise ValueError(
                    f"Expected normalized frame feature shape (126,), got {normalized_vec.shape}"
                )

            frame_features_list.append(normalized_vec)

        # Stack frame features into a sequence of shape (target_len, 126)
        sequence = np.stack(frame_features_list).astype(np.float32)

        # Ensure the final output shape is consistent and correct
        expected_shape = (target_len, 126)
        if sequence.shape != expected_shape:
            logger.error("Landmark sequence shape is incorrect: %s", sequence.shape)
            raise ValueError(
                f"Unexpected landmark sequence shape. Expected {expected_shape}, got {sequence.shape}"
            )

        return sequence

    def build_samples(self, samples: Iterable[DatasetSample]) -> dict[str, np.ndarray]:
        """Processes multiple dataset samples and returns a dictionary of features.

        Parameters
        ----------
        samples : Iterable of DatasetSample
            The collection of samples to process.

        Returns
        -------
        dict of str to np.ndarray
            A dictionary mapping sample_id to its corresponding landmark sequence.
        """
        if samples is None:
            logger.error("samples argument is None")
            raise TypeError("samples must be an iterable collection of DatasetSample")

        results = {}
        for sample in samples:
            results[sample.sample_id] = self.build_sample(sample)
        return results
