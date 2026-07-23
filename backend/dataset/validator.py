"""Dataset validator module for verifying dataset integrity and image correctness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import zipfile
from PIL import Image

from backend.dataset.exceptions import ValidationError

# Set up logging for the dataset validator
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Represents the detailed report generated after a dataset validation run.

    Attributes
    ----------
    is_valid : bool
        True if no invalid files or empty directories were found, False otherwise.
    total_files : int
        The total number of image files checked during validation.
    invalid_files : list of Path
        A list of paths to corrupted or invalid files detected.
    empty_dirs : list of Path
        A list of paths to empty directories detected.
    """

    is_valid: bool
    total_files: int
    invalid_files: list[Path]
    empty_dirs: list[Path]


class DatasetValidator:
    """Validates the directory structure, file integrity, and correctness of image datasets."""

    def __init__(self) -> None:
        """Initializes the DatasetValidator with an empty tracking list for invalid files."""
        self._invalid_files: list[Path] = []

    def validate(self, dataset_root: Path) -> ValidationReport:
        """Validates the dataset directory structure and image file integrity.

        Checks that the directory exists, recursively discovers image files,
        ignores hidden files/folders, detects empty directories, and identifies
        corrupted images.

        Parameters
        ----------
        dataset_root : Path
            The root directory of the dataset to validate.

        Returns
        -------
        ValidationReport
            The validation report containing status, total checked files,
            invalid files, and empty directories.

        Raises
        ------
        TypeError
            If dataset_root is not a pathlib.Path instance.
        ValidationError
            If the dataset_root does not exist or is not a directory.
        """
        if not isinstance(dataset_root, Path):
            logger.error("dataset_root must be a pathlib.Path instance, got: %s", type(dataset_root))
            raise TypeError("dataset_root must be a pathlib.Path instance")

        if not dataset_root.exists():
            logger.error("Dataset root directory does not exist: %s", dataset_root)
            raise ValidationError(f"Dataset root directory does not exist: {dataset_root}")

        if not dataset_root.is_dir():
            logger.error("Dataset root path is not a directory: %s", dataset_root)
            raise ValidationError(f"Dataset root is not a directory: {dataset_root}")

        self._invalid_files = []
        empty_dirs: list[Path] = []
        total_images_checked = 0

        supported_extensions = {".jpg", ".jpeg", ".png"}

        logger.info("Starting dataset validation at: %s", dataset_root)

        # Walk through the directory tree
        try:
            for path in dataset_root.rglob("*"):
                # Ignore hidden files and directories
                if self._is_hidden_or_in_hidden(path, dataset_root):
                    continue

                if path.is_file():
                    if path.suffix.lower() in supported_extensions:
                        total_images_checked += 1
                        if not self._is_valid_image(path):
                            self._invalid_files.append(path)
                elif path.is_dir():
                    # Check if the directory has any non-hidden files recursively
                    has_files = False
                    try:
                        for child in path.rglob("*"):
                            if child.is_file() and not self._is_hidden_or_in_hidden(child, dataset_root):
                                has_files = True
                                break
                    except OSError as e:
                        logger.error("Error scanning subdirectory %s: %s", path, e)
                        raise ValidationError(f"Error scanning subdirectory {path}: {e}") from e

                    if not has_files:
                        empty_dirs.append(path)

            # Check if dataset_root itself is empty
            has_root_files = False
            for child in dataset_root.rglob("*"):
                if child.is_file() and not self._is_hidden_or_in_hidden(child, dataset_root):
                    has_root_files = True
                    break
            if not has_root_files and dataset_root not in empty_dirs:
                empty_dirs.append(dataset_root)

        except OSError as e:
            logger.error("Fatal OS error during dataset validation: %s", e)
            raise ValidationError(f"Fatal error during dataset validation: {e}") from e

        is_valid = len(self._invalid_files) == 0 and len(empty_dirs) == 0

        logger.info(
            "Validation completed. Valid: %s. Checked images: %d. Invalid: %d. Empty dirs: %d.",
            is_valid,
            total_images_checked,
            len(self._invalid_files),
            len(empty_dirs),
        )

        return ValidationReport(
            is_valid=is_valid,
            total_files=total_images_checked,
            invalid_files=list(self._invalid_files),
            empty_dirs=sorted(empty_dirs),
        )

    def validate_archive(self, path: Path) -> bool:
        """Validates a zip archive's structural and file integrity.

        Parameters
        ----------
        path : Path
            The path to the ZIP archive.

        Returns
        -------
        bool
            True if the archive is valid and intact, False otherwise.

        Raises
        ------
        TypeError
            If path is not a pathlib.Path instance.
        ValidationError
            If a fatal OS error occurs during validation.
        """
        if not isinstance(path, Path):
            logger.error("path must be a pathlib.Path instance, got: %s", type(path))
            raise TypeError("path must be a pathlib.Path instance")

        if not path.exists() or not path.is_file():
            logger.warning("Archive path does not exist or is not a file: %s", path)
            return False

        if not zipfile.is_zipfile(path):
            logger.warning("Path is not a valid ZIP file: %s", path)
            return False

        try:
            with zipfile.ZipFile(path, "r") as z:
                # testzip returns the name of the first corrupted file or None if all are OK
                bad_file = z.testzip()
                if bad_file is not None:
                    logger.warning("Corrupted file %s found inside ZIP: %s", bad_file, path)
                    return False
            return True
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning("Error validating ZIP archive %s: %s", path, e)
            # If it's an OS permission/disk error, we treat it as a fatal ValidationError
            if isinstance(e, OSError) and not isinstance(e, zipfile.BadZipFile):
                logger.error("Fatal OS error during archive validation of %s: %s", path, e)
                raise ValidationError(f"Fatal error validating ZIP archive {path}: {e}") from e
            return False

    def list_invalid_files(self) -> list[Path]:
        """Returns the list of invalid files detected in the most recent validation.

        Returns
        -------
        list of Path
            A list of invalid file paths.
        """
        return list(self._invalid_files)

    def _is_hidden_or_in_hidden(self, path: Path, root: Path) -> bool:
        """Checks if a path is hidden (starts with '.') or lies inside a hidden folder relative to root."""
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        return any(part.startswith(".") for part in rel.parts)

    def _is_valid_image(self, file_path: Path) -> bool:
        """Verifies if an image file is correct and not corrupted using Pillow.

        Parameters
        ----------
        file_path : Path
            The path to the image file.

        Returns
        -------
        bool
            True if the image is valid and successfully verified/loaded, False otherwise.
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            # Re-open because verify() closes the file pointer and limits further operations
            with Image.open(file_path) as img:
                img.load()
            return True
        except Exception as e:
            logger.warning("Corrupted or invalid image detected at %s: %s", file_path, e)
            return False
