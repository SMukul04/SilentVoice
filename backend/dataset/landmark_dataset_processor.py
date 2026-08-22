from __future__ import annotations

from collections.abc import Iterable
import json
import logging
from pathlib import Path
from typing import Protocol, Any
import numpy as np

from backend.dataset.indexer import DatasetSample
from backend.dataset.splitter import DatasetSplit

# Set up logging
logger = logging.getLogger(__name__)


class IndexerProtocol(Protocol):
    """Protocol defining the interface for a dataset indexer."""

    def build_index(self) -> None:
        """Scans and indexes the dataset."""
        ...

    def get_num_samples(self) -> int:
        """Returns the number of samples."""
        ...

    def get_samples(self) -> list[DatasetSample]:
        """Returns the list of samples."""
        ...

    def get_num_classes(self) -> int:
        """Returns the number of classes."""
        ...

    def get_class_to_index(self) -> dict[str, int]:
        """Returns class to index mapping."""
        ...

    def get_index_to_class(self) -> dict[int, str]:
        """Returns index to class mapping."""
        ...


class SplitterProtocol(Protocol):
    """Protocol defining the interface for a dataset splitter."""

    random_seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def split(self, samples: Iterable[DatasetSample]) -> DatasetSplit:
        """Splits the samples into splits."""
        ...


class SamplerProtocol(Protocol):
    """Protocol defining the interface for a frame sampler."""

    sequence_length: int


class BuilderProtocol(Protocol):
    """Protocol defining the interface for a landmark builder."""

    frame_sampler: SamplerProtocol

    def build_sample(self, sample: DatasetSample) -> np.ndarray:
        """Converts sample to sequence of landmarks."""
        ...


class LandmarkDatasetProcessor:
    """Orchestrates indexing, splitting, landmark sequence extraction, and compressed storage."""

    def __init__(
        self,
        indexer: IndexerProtocol,
        splitter: SplitterProtocol,
        builder: BuilderProtocol,
        output_dir: Path,
    ) -> None:
        """Initializes the LandmarkDatasetProcessor with dependent components.

        Parameters
        ----------
        indexer : IndexerProtocol
            The dataset indexer component.
        splitter : SplitterProtocol
            The dataset splitter component.
        builder : BuilderProtocol
            The landmark builder component.
        output_dir : Path
            The directory path where output files should be saved.

        Raises
        ------
        TypeError
            If any dependency component has an invalid type.
        ValueError
            If output_dir exists but is not a directory.
        """
        # Validate indexer duck typing contract
        for attr in ["build_index", "get_num_samples", "get_samples", "get_num_classes", "get_class_to_index", "get_index_to_class"]:
            if not hasattr(indexer, attr) or not callable(getattr(indexer, attr)):
                raise TypeError(f"indexer must provide a callable '{attr}' method")

        # Validate splitter duck typing contract
        if not hasattr(splitter, "split") or not callable(getattr(splitter, "split")):
            raise TypeError("splitter must provide a callable 'split' method")
        for attr in ["random_seed", "train_ratio", "validation_ratio", "test_ratio"]:
            if not hasattr(splitter, attr):
                raise TypeError(f"splitter must provide attribute '{attr}'")

        # Validate builder duck typing contract
        if not hasattr(builder, "build_sample") or not callable(getattr(builder, "build_sample")):
            raise TypeError("builder must provide a callable 'build_sample' method")
        if not hasattr(builder, "frame_sampler") or not hasattr(builder.frame_sampler, "sequence_length"):
            raise TypeError("builder must have a frame_sampler with a sequence_length attribute")

        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path instance")

        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output_dir exists but is not a directory: {output_dir}")


        self.indexer = indexer
        self.splitter = splitter
        self.builder = builder
        self.output_dir = output_dir

        logger.info("LandmarkDatasetProcessor successfully initialized.")

    def build_index(self) -> None:
        """Scans the dataset directory using the indexer component.

        Raises
        ------
        ValueError
            If the indexer finds no samples.
        """
        logger.info("Building dataset index using indexer...")
        self.indexer.build_index()
        if self.indexer.get_num_samples() == 0:
            raise ValueError("Dataset is empty. No video samples discovered.")

    def split_dataset(self) -> DatasetSplit:
        """Splits the indexed dataset samples using the splitter component.

        Returns
        -------
        DatasetSplit
            The partitioned dataset splits.
        """
        logger.info("Splitting dataset using splitter...")
        samples = self.indexer.get_samples()
        return self.splitter.split(samples)

    def process_split(self, samples: list[DatasetSample], split_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Processes samples in a split, converting each to features and labels.

        Parameters
        ----------
        samples : list of DatasetSample
            The list of samples inside the split.
        split_name : str
            The name of the split (for logging/printing progress).

        Returns
        -------
        tuple of np.ndarray
            X array of shape (num_samples, sequence_length, feature_dimension) and
            y array of shape (num_samples,).

        Raises
        ------
        ValueError
            If the split is empty or a sample returns an inconsistent shape.
        """
        if not samples:
            raise ValueError(f"Cannot process an empty split for: {split_name}")

        print(f"Processing {split_name} dataset...")
        X_list = []
        y_list = []

        total = len(samples)
        expected_seq_len = self.builder.frame_sampler.sequence_length
        expected_feat_dim = 126

        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{total}] Processing sample: {sample.sample_id}")
            # Build landmark feature sequence
            sequence = self.builder.build_sample(sample)

            if sequence.shape != (expected_seq_len, expected_feat_dim):
                logger.error(
                    "Sample %s returned sequence with shape %s, expected %s",
                    sample.sample_id,
                    sequence.shape,
                    (expected_seq_len, expected_feat_dim),
                )
                raise ValueError(
                    f"Sample {sample.sample_id} returned inconsistent shape {sequence.shape}. "
                    f"Expected {(expected_seq_len, expected_feat_dim)}"
                )

            X_list.append(sequence)
            y_list.append(sample.class_index)

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        expected_X_shape = (total, expected_seq_len, expected_feat_dim)
        expected_y_shape = (total,)

        if X.shape != expected_X_shape:
            raise ValueError(
                f"Stacked feature array shape is {X.shape}, expected {expected_X_shape}"
            )
        if y.shape != expected_y_shape:
            raise ValueError(
                f"Stacked label array shape is {y.shape}, expected {expected_y_shape}"
            )

        return X, y

    def save_split(self, X: np.ndarray, y: np.ndarray, filename: str) -> None:
        """Saves feature and label arrays as a compressed .npz file.

        Parameters
        ----------
        X : np.ndarray
            Features array.
        y : np.ndarray
            Labels array.
        filename : str
            The name of the target file.

        Raises
        ------
        IOError
            If saving to disk fails.
        """
        output_path = self.output_dir / filename
        logger.info("Saving split to %s", output_path)
        try:
            np.savez_compressed(output_path, X=X, y=y)
        except Exception as e:
            logger.error("Failed to save split to %s: %s", output_path, e)
            raise IOError(f"Failed to save split to {output_path}: {e}") from e

    def save_metadata(self, split_result: DatasetSplit) -> None:
        """Saves preprocessing and split description metadata to metadata.json.

        Parameters
        ----------
        split_result : DatasetSplit
            The partitioned dataset split containing count metadata.

        Raises
        ------
        IOError
            If saving to disk fails.
        """
        metadata_path = self.output_dir / "metadata.json"
        logger.info("Saving metadata to %s", metadata_path)

        metadata = {
            "num_classes": self.indexer.get_num_classes(),
            "class_to_index": self.indexer.get_class_to_index(),
            "index_to_class": {str(k): v for k, v in self.indexer.get_index_to_class().items()},
            "sequence_length": self.builder.frame_sampler.sequence_length,
            "feature_dimension": 126,
            "train_sample_count": split_result.train_count,
            "validation_sample_count": split_result.validation_count,
            "test_sample_count": split_result.test_count,
            "random_seed": self.splitter.random_seed,
            "split_ratios": {
                "train": self.splitter.train_ratio,
                "validation": self.splitter.validation_ratio,
                "test": self.splitter.test_ratio,
            },
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error("Failed to save metadata to %s: %s", metadata_path, e)
            raise IOError(f"Failed to save metadata to {metadata_path}: {e}") from e

    def process(self) -> None:
        """Executes the complete landmark dataset preprocessing and storage pipeline.

        1. Creates/validates output directory.
        2. Indexes dataset.
        3. Splits dataset deterministically.
        4. Processes each split.
        5. Saves processed splits as compressed numpy arrays (.npz).
        6. Saves metadata descriptor file (metadata.json).
        """
        # Ensure output directory exists and is valid
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {self.output_dir}")
        else:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise IOError(f"Failed to create output directory {self.output_dir}: {e}") from e

        # Indexing and Splitting
        self.build_index()
        split_result = self.split_dataset()

        # Process splits
        X_train, y_train = self.process_split(split_result.train, "train")
        X_val, y_val = self.process_split(split_result.validation, "validation")
        X_test, y_test = self.process_split(split_result.test, "test")

        # Save splits
        self.save_split(X_train, y_train, "train.npz")
        self.save_split(X_val, y_val, "validation.npz")
        self.save_split(X_test, y_test, "test.npz")

        # Save metadata
        self.save_metadata(split_result)
        logger.info("Complete preprocessing and storage process finished successfully.")
