"""Dataset extractor module for extracting downloaded ZIP archives for the SilentVoice project."""

from __future__ import annotations

import logging
from pathlib import Path
import zipfile

from backend.dataset.exceptions import ExtractionError

# Set up logging for the dataset extractor
logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Handles the extraction of downloaded dataset ZIP archives.

    Extracts archives to a designated raw data directory (defaulting to
    'datasets/raw/'), maintaining folder hierarchy and preventing duplicate
    extractions.
    """

    def __init__(self, output_dir: Path = Path("datasets/raw")) -> None:
        """Initializes the DatasetExtractor.

        Parameters
        ----------
        output_dir : Path, optional
            The destination directory where ZIP archives will be extracted.
            Defaults to Path("datasets/raw").

        Raises
        ------
        TypeError
            If output_dir is not a pathlib.Path instance.
        """
        if not isinstance(output_dir, Path):
            logger.error("output_dir must be a pathlib.Path instance, got: %s", type(output_dir))
            raise TypeError("output_dir must be a pathlib.Path instance")

        self.output_dir = output_dir

        # Ensure target extraction directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Output directory configured: %s", self.output_dir)
        except OSError as e:
            logger.error("Failed to create output directory %s: %s", self.output_dir, e)
            raise ExtractionError(f"Could not create output directory: {self.output_dir}") from e

    def extract(self, zip_path: Path) -> Path:
        """Extracts a single ZIP archive to the output directory.

        If the archive is already extracted (determined by checking the presence
        of its files), extraction is skipped.

        Parameters
        ----------
        zip_path : Path
            The path to the ZIP archive to extract.

        Returns
        -------
        Path
            The directory path where the ZIP contents were extracted.

        Raises
        ------
        TypeError
            If zip_path is not a pathlib.Path instance.
        FileNotFoundError
            If the zip_path file does not exist.
        ExtractionError
            If the ZIP file is invalid, corrupted, contains path traversal
            vulnerabilities, or if extraction fails.
        """
        if not isinstance(zip_path, Path):
            logger.error("zip_path must be a pathlib.Path instance, got: %s", type(zip_path))
            raise TypeError("zip_path must be a pathlib.Path instance")

        if not zip_path.exists():
            logger.error("ZIP archive not found: %s", zip_path)
            raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

        # Check if already extracted
        if self.is_extracted(zip_path):
            logger.info("Archive %s is already extracted. Skipping.", zip_path.name)
            return self.output_dir

        logger.info("Extracting archive: %s to %s", zip_path, self.output_dir)
        resolved_output_dir = self.output_dir.resolve()

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                # Security Check: Prevent Zip Slip (Path Traversal)
                for member in z.infolist():
                    # Resolve target path and verify it lies inside output_dir
                    target_path = Path(self.output_dir / member.filename).resolve()
                    if resolved_output_dir not in target_path.parents and target_path != resolved_output_dir:
                        logger.error(
                            "Path traversal vulnerability detected in ZIP archive %s: file %s",
                            zip_path,
                            member.filename,
                        )
                        raise ExtractionError(
                            f"Path traversal vulnerability detected in ZIP: {member.filename}"
                        )

                # Extract the files using built-in extractall (streams contents directly to disk)
                z.extractall(self.output_dir)
                logger.info("Successfully extracted archive %s", zip_path.name)

        except zipfile.BadZipFile as e:
            logger.error("Invalid or corrupted ZIP file %s: %s", zip_path, e)
            raise ExtractionError(f"Invalid or corrupted ZIP file {zip_path}: {e}") from e
        except OSError as e:
            logger.error("OS error occurred during extraction of %s: %s", zip_path, e)
            raise ExtractionError(f"Failed to extract ZIP file {zip_path}: {e}") from e

        return self.output_dir

    def extract_all(self, directory: Path) -> list[Path]:
        """Finds and extracts all ZIP archives located in the specified directory.

        Parameters
        ----------
        directory : Path
            The directory to search for ZIP archives.

        Returns
        -------
        list of Path
            A list containing the destination directory paths where each archive was extracted.

        Raises
        ------
        TypeError
            If directory is not a pathlib.Path instance.
        FileNotFoundError
            If the specified directory does not exist.
        ExtractionError
            If any error occurs during list or extraction operations.
        """
        if not isinstance(directory, Path):
            logger.error("directory must be a pathlib.Path instance, got: %s", type(directory))
            raise TypeError("directory must be a pathlib.Path instance")

        if not directory.exists():
            logger.error("Directory not found: %s", directory)
            raise FileNotFoundError(f"Directory not found: {directory}")

        archives = self.list_archives(directory)
        logger.info("Found %d archives to extract in %s", len(archives), directory)

        extracted_paths: list[Path] = []
        for archive in archives:
            try:
                dest = self.extract(archive)
                extracted_paths.append(dest)
            except (ExtractionError, FileNotFoundError) as e:
                logger.error("Failed to extract %s during extract_all: %s", archive, e)
                raise

        return extracted_paths

    def is_extracted(self, zip_path: Path) -> bool:
        """Checks if all the files in the ZIP archive have already been extracted.

        Parameters
        ----------
        zip_path : Path
            The path to the ZIP archive.

        Returns
        -------
        bool
            True if all files (excluding directories) exist in the output
            directory, False otherwise. Returns False if the zip_path does
            not exist or is invalid.
        """
        if not isinstance(zip_path, Path):
            logger.error("zip_path must be a pathlib.Path instance, got: %s", type(zip_path))
            return False

        if not zip_path.exists():
            return False

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                members = z.infolist()
                if not members:
                    # If empty zip archive, consider it not extracted or handle gracefully
                    return False

                for member in members:
                    if member.is_dir():
                        continue
                    target_path = self.output_dir / member.filename
                    if not target_path.exists():
                        return False
            return True
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning("Error checking extraction status of %s: %s", zip_path, e)
            return False

    def list_archives(self, directory: Path) -> list[Path]:
        """Lists all ZIP archives (*.zip) in the specified directory.

        Parameters
        ----------
        directory : Path
            The directory to scan for ZIP archives.

        Returns
        -------
        list of Path
            A sorted list of Path objects pointing to the found ZIP files.

        Raises
        ------
        TypeError
            If directory is not a pathlib.Path instance.
        FileNotFoundError
            If the specified directory does not exist or is not a directory.
        """
        if not isinstance(directory, Path):
            logger.error("directory must be a pathlib.Path instance, got: %s", type(directory))
            raise TypeError("directory must be a pathlib.Path instance")

        if not directory.exists() or not directory.is_dir():
            logger.error("Directory not found or is not a directory: %s", directory)
            raise FileNotFoundError(f"Directory not found or is not a directory: {directory}")

        archives = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
        return sorted(archives)
