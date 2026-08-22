"""Dataset splitter module for deterministic, class-aware partitioning of samples."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
import math
import random

from backend.dataset.indexer import DatasetSample

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Dataclass holding the training, validation, and test sample splits.

    Attributes
    ----------
    train : list of DatasetSample
        The training set samples.
    validation : list of DatasetSample
        The validation set samples.
    test : list of DatasetSample
        The test set samples.
    """

    train: list[DatasetSample]
    validation: list[DatasetSample]
    test: list[DatasetSample]

    @property
    def train_count(self) -> int:
        """Returns the number of training samples."""
        return len(self.train)

    @property
    def validation_count(self) -> int:
        """Returns the number of validation samples."""
        return len(self.validation)

    @property
    def test_count(self) -> int:
        """Returns the number of test samples."""
        return len(self.test)

    @property
    def total_count(self) -> int:
        """Returns the total number of samples across all splits."""
        return len(self.train) + len(self.validation) + len(self.test)


class DatasetSplitter:
    """Splits a dataset deterministically and in a class-aware manner."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        """Initializes the DatasetSplitter with ratios and a random seed.

        Parameters
        ----------
        train_ratio : float, default 0.70
            The proportion of samples for training.
        validation_ratio : float, default 0.15
            The proportion of samples for validation.
        test_ratio : float, default 0.15
            The proportion of samples for testing.
        random_seed : int, default 42
            The seed used for deterministic shuffling.

        Raises
        ------
        TypeError
            If ratios are not numeric or random_seed is not an integer.
        ValueError
            If ratios are not positive or do not sum to approximately 1.0.
        """
        # Validate random seed (explicitly check for bool since it's a subclass of int)
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            logger.error("random_seed must be an integer, got: %s", type(random_seed))
            raise TypeError("random_seed must be an integer")

        # Validate ratios are numeric and positive
        for ratio_name, ratio_val in [
            ("train_ratio", train_ratio),
            ("validation_ratio", validation_ratio),
            ("test_ratio", test_ratio),
        ]:
            if not isinstance(ratio_val, (int, float)) or isinstance(ratio_val, bool):
                logger.error("%s must be numeric, got: %s", ratio_name, type(ratio_val))
                raise TypeError(f"{ratio_name} must be numeric")
            if ratio_val <= 0:
                logger.error("%s must be greater than 0, got: %f", ratio_name, ratio_val)
                raise ValueError(f"{ratio_name} must be greater than 0")

        # Validate ratios sum to 1.0 (allowing small float tolerance)
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not math.isclose(total_ratio, 1.0, abs_tol=1e-9):
            logger.error("Ratios must sum to 1.0, got: %f", total_ratio)
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        logger.debug(
            "Initialized DatasetSplitter: train=%f, val=%f, test=%f, seed=%d",
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
            self.random_seed,
        )

    def split(self, samples: Iterable[DatasetSample]) -> DatasetSplit:
        """Deterministically splits the input samples into train, validation, and test sets.

        Parameters
        ----------
        samples : Iterable of DatasetSample
            The collection of dataset samples to split.

        Returns
        -------
        DatasetSplit
            A result container containing the lists of samples for each split.

        Raises
        ------
        ValueError
            If the samples iterable is empty.
        TypeError
            If the samples argument is not iterable.
        """
        if samples is None:
            logger.error("samples argument is None")
            raise TypeError("samples must be an iterable collection of DatasetSample")

        try:
            samples_list = list(samples)
        except TypeError as e:
            logger.error("samples argument is not iterable: %s", e)
            raise TypeError("samples must be an iterable collection of DatasetSample") from e

        if not samples_list:
            logger.error("Received empty samples list for split")
            raise ValueError("samples collection cannot be empty")

        # Group samples by their class names
        class_groups: dict[str, list[DatasetSample]] = {}
        for sample in samples_list:
            # Basic validation of expected sample attributes
            if not hasattr(sample, "class_name") or not hasattr(sample, "sample_id"):
                logger.error("Input sample object is missing required attributes (class_name, sample_id)")
                raise TypeError("Sample objects must have 'class_name' and 'sample_id' attributes")
            class_groups.setdefault(sample.class_name, []).append(sample)

        train_list: list[DatasetSample] = []
        val_list: list[DatasetSample] = []
        test_list: list[DatasetSample] = []

        # Create a deterministic random number generator using the configured seed
        rng = random.Random(self.random_seed)

        # Process each class group alphabetically to guarantee execution path determinism
        for class_name in sorted(class_groups.keys()):
            group = class_groups[class_name]

            # Sort the group by sample_id to establish a baseline deterministic ordering.
            # This ensures that even if the input samples are ordered differently,
            # the deterministic shuffle yields the exact same split lists.
            sorted_group = sorted(group, key=lambda s: s.sample_id)

            # Shuffle the class list deterministically
            rng.shuffle(sorted_group)

            # Partition using cumulative rounded boundaries to handle small classes cleanly
            n_samples = len(sorted_group)
            idx_train = int(round(n_samples * self.train_ratio))
            idx_val = int(round(n_samples * (self.train_ratio + self.validation_ratio)))

            train_list.extend(sorted_group[:idx_train])
            val_list.extend(sorted_group[idx_train:idx_val])
            test_list.extend(sorted_group[idx_val:])

        logger.info(
            "Split completed. Total input: %d. Train: %d, Val: %d, Test: %d",
            len(samples_list),
            len(train_list),
            len(val_list),
            len(test_list),
        )

        return DatasetSplit(
            train=train_list,
            validation=val_list,
            test=test_list,
        )
