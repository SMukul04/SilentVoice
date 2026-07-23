"""Dataset frame extractor module for converting videos into sequential JPEG frame folders."""

from __future__ import annotations

import logging
from pathlib import Path
import cv2

from backend.dataset.exceptions import ExtractionError, ValidationError

# Set up logging for the frame extractor
logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts video frames into sequential JPEGs while preserving directory structures."""

    def extract_video(self, video_path: Path, output_dir: Path) -> int:
        """Extracts all frames from a video file and saves them as JPEG images.

        Parameters
        ----------
        video_path : Path
            The path to the source video file.
        output_dir : Path
            The destination directory where the JPEG frames will be saved.

        Returns
        -------
        int
            The total number of frames successfully extracted.

        Raises
        ------
        TypeError
            If video_path or output_dir is not a pathlib.Path instance.
        ValidationError
            If the video_path does not exist or is not a file.
        ExtractionError
            If the video file cannot be opened, has no frames, or if saving frames fails.
        """
        if not isinstance(video_path, Path):
            logger.error("video_path must be a pathlib.Path instance, got: %s", type(video_path))
            raise TypeError("video_path must be a pathlib.Path instance")

        if not isinstance(output_dir, Path):
            logger.error("output_dir must be a pathlib.Path instance, got: %s", type(output_dir))
            raise TypeError("output_dir must be a pathlib.Path instance")

        if not video_path.exists():
            logger.error("Video file does not exist: %s", video_path)
            raise ValidationError(f"Video file does not exist: {video_path}")

        if not video_path.is_file():
            logger.error("Video path is not a file: %s", video_path)
            raise ValidationError(f"Video path is not a file: {video_path}")

        # Open the video using cv2.VideoCapture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video file: %s", video_path)
            raise ExtractionError(f"Could not open video file: {video_path}")

        # Ensure target output directory exists
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            cap.release()
            logger.error("Failed to create output directory %s: %s", output_dir, e)
            raise ExtractionError(f"Failed to create output directory {output_dir}: {e}") from e

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_name = f"{frame_count:06d}.jpg"
                frame_path = output_dir / frame_name

                # Save the frame image using cv2.imwrite with JPEG quality = 95
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    raise ExtractionError(f"Failed to write frame image: {frame_path}")

            if frame_count == 0:
                logger.error("No frames were extracted from video: %s (empty or corrupted video)", video_path)
                raise ExtractionError(f"No frames were extracted from video: {video_path}")

            logger.info("Successfully extracted %d frames from %s to %s", frame_count, video_path.name, output_dir)

        except Exception as e:
            logger.error("Error occurred during extraction of %s: %s", video_path, e)
            if not isinstance(e, ExtractionError):
                raise ExtractionError(f"Failed to extract frames from {video_path}: {e}") from e
            raise
        finally:
            cap.release()

        return frame_count

    def extract_directory(self, dataset_root: Path, output_root: Path) -> dict[str, int]:
        """Recursively processes all supported videos in dataset_root and extracts their frames.

        Parameters
        ----------
        dataset_root : Path
            The root directory containing raw video folders.
        output_root : Path
            The root directory where processed frame folders will be stored.

        Returns
        -------
        dict of str to int
            A dictionary containing stats with keys: "processed", "skipped", "failed".

        Raises
        ------
        TypeError
            If dataset_root or output_root is not a pathlib.Path instance.
        ValidationError
            If the dataset_root does not exist or is not a directory.
        """
        if not isinstance(dataset_root, Path):
            logger.error("dataset_root must be a pathlib.Path instance, got: %s", type(dataset_root))
            raise TypeError("dataset_root must be a pathlib.Path instance")

        if not isinstance(output_root, Path):
            logger.error("output_root must be a pathlib.Path instance, got: %s", type(output_root))
            raise TypeError("output_root must be a pathlib.Path instance")

        if not dataset_root.exists():
            logger.error("dataset_root directory does not exist: %s", dataset_root)
            raise ValidationError(f"dataset_root directory does not exist: {dataset_root}")

        if not dataset_root.is_dir():
            logger.error("dataset_root path is not a directory: %s", dataset_root)
            raise ValidationError(f"dataset_root is not a directory: {dataset_root}")

        videos = self.list_supported_videos(dataset_root)
        logger.info("Starting extraction... Found %d videos in %s", len(videos), dataset_root)

        stats = {"processed": 0, "skipped": 0, "failed": 0}

        for video_path in videos:
            # Check if this video has already been processed
            if self.is_already_processed(video_path, output_root):
                logger.info("Skipping already processed video: %s", video_path.name)
                stats["skipped"] += 1
                continue

            # Compute relative path and target output directory
            try:
                rel_path = video_path.relative_to(dataset_root)
            except ValueError:
                # Fallback to general parsing if not directly relative
                rel_path = self._compute_fallback_relative_path(video_path)

            dest_dir = output_root / rel_path.parent / video_path.stem
            logger.info("Processing video %s...", video_path.name)

            try:
                self.extract_video(video_path, dest_dir)
                stats["processed"] += 1
            except Exception as e:
                logger.error("Failed to extract frames for video %s: %s", video_path, e)
                stats["failed"] += 1

        logger.info("Finished extraction. Stats: %s", stats)
        return stats

    def list_supported_videos(self, directory: Path) -> list[Path]:
        """Lists all supported video files recursively within a directory, ignoring hidden items.

        Parameters
        ----------
        directory : Path
            The directory to search for videos.

        Returns
        -------
        list of Path
            A sorted list of Paths to the discovered supported videos.

        Raises
        ------
        TypeError
            If directory is not a pathlib.Path instance.
        ValidationError
            If the directory does not exist or is not a directory.
        """
        if not isinstance(directory, Path):
            logger.error("directory must be a pathlib.Path instance, got: %s", type(directory))
            raise TypeError("directory must be a pathlib.Path instance")

        if not directory.exists():
            logger.error("Directory does not exist: %s", directory)
            raise ValidationError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            logger.error("Directory is not a directory: %s", directory)
            raise ValidationError(f"Directory is not a directory: {directory}")

        supported_suffixes = {".mov", ".mp4", ".avi", ".mkv"}
        videos: list[Path] = []

        try:
            for p in directory.rglob("*"):
                # Ignore hidden files/folders (starting with dot)
                try:
                    rel = p.relative_to(directory)
                    if any(part.startswith(".") for part in rel.parts):
                        continue
                except ValueError:
                    if any(part.startswith(".") for part in p.parts):
                        continue

                if p.is_file() and p.suffix.lower() in supported_suffixes:
                    videos.append(p)
        except OSError as e:
            logger.error("OS error scanning directory %s: %s", directory, e)
            raise ValidationError(f"OS error scanning directory: {e}") from e

        return sorted(videos)

    def is_already_processed(self, video_path: Path, output_root: Path) -> bool:
        """Checks if a video file has already been processed to the output root.

        Parameters
        ----------
        video_path : Path
            The path to the video file.
        output_root : Path
            The root directory for processed files.

        Returns
        -------
        bool
            True if the target folder exists and contains at least one JPG frame, False otherwise.

        Raises
        ------
        TypeError
            If video_path or output_root is not a pathlib.Path instance.
        """
        if not isinstance(video_path, Path):
            logger.error("video_path must be a pathlib.Path instance, got: %s", type(video_path))
            raise TypeError("video_path must be a pathlib.Path instance")

        if not isinstance(output_root, Path):
            logger.error("output_root must be a pathlib.Path instance, got: %s", type(output_root))
            raise TypeError("output_root must be a pathlib.Path instance")

        dest_dir = self._compute_output_directory(video_path, output_root)

        if not dest_dir.exists() or not dest_dir.is_dir():
            return False

        try:
            for p in dest_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".jpg":
                    return True
        except OSError:
            pass

        return False

    def _compute_output_directory(self, video_path: Path, output_root: Path) -> Path:
        """Computes the target output directory path for a given video and output root.

        Parameters
        ----------
        video_path : Path
            The path to the video file.
        output_root : Path
            The root directory for processed files.

        Returns
        -------
        Path
            The calculated target output directory.
        """
        parts = video_path.parts
        if "raw" in parts:
            idx = parts.index("raw")
            rel_parts = parts[idx + 1 :]
            rel_path = Path(*rel_parts)
        elif len(parts) >= 3:
            rel_path = Path(*parts[-3:])
        else:
            rel_path = Path(*parts)

        return output_root / rel_path.parent / video_path.stem

    def _compute_fallback_relative_path(self, video_path: Path) -> Path:
        """Computes a fallback relative path for a video when relative_to fails.

        Parameters
        ----------
        video_path : Path
            The path to the video file.

        Returns
        -------
        Path
            A calculated relative path structure.
        """
        parts = video_path.parts
        if "raw" in parts:
            idx = parts.index("raw")
            rel_parts = parts[idx + 1 :]
            return Path(*rel_parts)
        if len(parts) >= 3:
            return Path(*parts[-3:])
        return Path(*parts)
