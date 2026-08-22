"""Dataset indexer module for discovering and organizing SilentVoice dataset samples."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

from backend.dataset.exceptions import ValidationError

# Set up logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSample:
    """Immutable data model representing a single video dataset sample.

    Attributes
    ----------
    sample_id : str
        A unique, stable identifier for the sample (dataset-relative path).
    class_name : str
        The normalized class name of the sample.
    class_index : int
        The deterministic index assigned to the class.
    video_path : Path
        The absolute path to the video sample directory.
    frame_paths : tuple[Path, ...]
        Naturally sorted absolute paths to valid image frames.
    num_frames : int
        The number of frames in this sample.
    """

    sample_id: str
    class_name: str
    class_index: int
    video_path: Path
    frame_paths: tuple[Path, ...]
    num_frames: int


class DatasetIndexer:
    """Discovers and indexes dataset samples from the filesystem."""

    def __init__(self, dataset_root: Path) -> None:
        """Initializes the DatasetIndexer.

        Parameters
        ----------
        dataset_root : Path
            The root directory of the dataset.

        Raises
        ------
        TypeError
            If dataset_root is not a Path.
        ValidationError
            If the directory does not exist or is not a directory.
        """
        if not isinstance(dataset_root, Path):
            raise TypeError("dataset_root must be a pathlib.Path instance")
        if not dataset_root.exists():
            raise ValidationError(f"Dataset root directory does not exist: {dataset_root}")
        if not dataset_root.is_dir():
            raise ValidationError(f"Dataset root is not a directory: {dataset_root}")

        self.dataset_root = dataset_root
        self._samples: list[DatasetSample] = []
        self._class_to_index: dict[str, int] = {}
        self._index_to_class: dict[int, str] = {}

    def build_index(self) -> None:
        """Scans the dataset directory tree and builds the sample index.

        Can be safely called multiple times to rebuild the index from scratch.
        """
        # Reset internal state to ensure rebuild safety
        self._samples = []
        self._class_to_index = {}
        self._index_to_class = {}

        supported_extensions = {".jpg", ".jpeg", ".png"}
        video_to_frames: dict[Path, list[Path]] = {}

        logger.info("Scanning dataset root for valid image frames at: %s", self.dataset_root)

        try:
            for path in self.dataset_root.rglob("*"):
                # Ignore hidden files and directories
                if self._is_hidden(path):
                    continue

                if path.is_file() and path.suffix.lower() in supported_extensions:
                    parent = path.parent
                    if parent not in video_to_frames:
                        video_to_frames[parent] = []
                    video_to_frames[parent].append(path)
        except OSError as e:
            logger.error("OS error occurred while scanning dataset directory %s: %s", self.dataset_root, e)
            raise ValidationError(f"OS error scanning dataset directory: {e}") from e

        if not video_to_frames:
            logger.info("Build index completed. No valid samples found.")
            return

        # Map classes to their corresponding sample info
        class_to_samples: dict[str, list[tuple[Path, tuple[Path, ...]]]] = {}
        for video_path, frames in video_to_frames.items():
            # Natural sorting of frame paths
            sorted_frames = tuple(sorted(frames, key=self._natural_sort_key))

            # Determine the parent class folder of each sample
            class_dir = video_path.parent
            normalized_class = self._normalize_class_name(class_dir.name)

            if normalized_class not in class_to_samples:
                class_to_samples[normalized_class] = []
            class_to_samples[normalized_class].append((video_path, sorted_frames))

        # Assign class indices using alphabetically sorted normalized class names
        sorted_classes = sorted(class_to_samples.keys())
        self._class_to_index = {name: idx for idx, name in enumerate(sorted_classes)}
        self._index_to_class = {idx: name for idx, name in enumerate(sorted_classes)}

        # Build and collect DatasetSample objects
        for class_name in sorted_classes:
            class_index = self._class_to_index[class_name]
            for video_path, frame_paths in class_to_samples[class_name]:
                sample_id = video_path.relative_to(self.dataset_root).as_posix()
                sample = DatasetSample(
                    sample_id=sample_id,
                    class_name=class_name,
                    class_index=class_index,
                    video_path=video_path,
                    frame_paths=frame_paths,
                    num_frames=len(frame_paths),
                )
                self._samples.append(sample)

        # Produce a deterministic ordering of samples based on sample ID
        self._samples.sort(key=lambda s: s.sample_id)
        logger.info(
            "Index built successfully. Classes: %d, Samples: %d, Total Frames: %d",
            self.get_num_classes(),
            self.get_num_samples(),
            self.get_total_frames(),
        )

    def get_class_to_index(self) -> dict[str, int]:
        """Returns a copy of the mapping from class names to indices."""
        return dict(self._class_to_index)

    def get_index_to_class(self) -> dict[int, str]:
        """Returns a copy of the mapping from indices to class names."""
        return dict(self._index_to_class)

    def get_num_classes(self) -> int:
        """Returns the total number of classes discovered."""
        return len(self._class_to_index)

    def get_samples(self) -> list[DatasetSample]:
        """Returns a copy of the internal collection of samples."""
        return list(self._samples)

    def get_num_samples(self) -> int:
        """Returns the total number of samples discovered."""
        return len(self._samples)

    def get_total_frames(self) -> int:
        """Returns the sum of all frames across all samples."""
        return sum(sample.num_frames for sample in self._samples)

    def _is_hidden(self, path: Path) -> bool:
        """Checks if a path or any of its parent components relative to root is hidden."""
        try:
            rel = path.relative_to(self.dataset_root)
        except ValueError:
            rel = path
        return any(part.startswith(".") for part in rel.parts)

    def _normalize_class_name(self, name: str) -> str:
        """Normalizes class folder name to extract cleaner category names.

        Removes only the leading numeric prefix followed by a period and optional whitespace.
        E.g. "23. high" -> "high", "hello" -> "hello".
        """
        return re.sub(r"^\d+\.\s*", "", name)

    def _natural_sort_key(self, path: Path) -> list[str | int]:
        """Generates a key for natural sorting of path filenames containing numbers."""
        parts = re.split(r"(\d+)", path.name)
        return [int(text) if text.isdigit() else text.lower() for text in parts]
