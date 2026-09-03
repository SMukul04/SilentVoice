"""Unit tests for the DatasetValidator class."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

from backend.dataset.exceptions import ValidationError
from backend.dataset.validator import DatasetValidator


class TestDatasetValidator(unittest.TestCase):
    """Unit tests to verify functionality and correctness of DatasetValidator."""

    def setUp(self) -> None:
        """Sets up temporary directories and validator instance."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.validator = DatasetValidator()

    def tearDown(self) -> None:
        """Cleans up temporary directories."""
        self.temp_dir.cleanup()

    def create_mock_image(self, path: Path, fmt: str = "JPEG") -> None:
        """Helper to create a valid mock image at the specified path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (10, 10), color="blue")
        img.save(path, format=fmt)

    def create_corrupted_file(self, path: Path) -> None:
        """Helper to create a corrupted image file (invalid bytes)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"not an image file content, just garbage bytes")

    def test_validate_nonexistent_directory(self) -> None:
        """Tests that validate raises ValidationError when the path does not exist."""
        ghost_path = self.temp_dir_path / "ghost_folder"
        with self.assertRaises(ValidationError) as ctx:
            self.validator.validate(ghost_path)
        self.assertIn("does not exist", str(ctx.exception))

    def test_validate_not_a_directory(self) -> None:
        """Tests that validate raises ValidationError when path is a file instead of a directory."""
        file_path = self.temp_dir_path / "regular_file.txt"
        file_path.touch()
        with self.assertRaises(ValidationError) as ctx:
            self.validator.validate(file_path)
        self.assertIn("is not a directory", str(ctx.exception))

    def test_validate_invalid_type(self) -> None:
        """Tests that passing an invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            self.validator.validate("string_path")  # type: ignore

    def test_validate_success_clean_dataset(self) -> None:
        """Tests successful validation of a clean dataset containing images."""
        # Create a valid structure
        self.create_mock_image(self.temp_dir_path / "class1" / "img1.jpg", "JPEG")
        self.create_mock_image(self.temp_dir_path / "class1" / "img2.png", "PNG")
        self.create_mock_image(self.temp_dir_path / "class2" / "img3.jpeg", "JPEG")

        # Add some ignored non-hidden files like text files or videos
        (self.temp_dir_path / "class1" / "readme.txt").touch()

        # Add hidden folder that should be ignored
        hidden_dir = self.temp_dir_path / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config").touch()

        # Run validation
        report = self.validator.validate(self.temp_dir_path)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_files, 3)
        self.assertEqual(len(report.invalid_files), 0)
        self.assertEqual(len(report.empty_dirs), 0)

    def test_validate_detects_empty_directories(self) -> None:
        """Tests that empty directories are detected."""
        # Create normal class with files
        self.create_mock_image(self.temp_dir_path / "class_ok" / "img.jpg", "JPEG")

        # Create empty directory
        empty_dir = self.temp_dir_path / "class_empty"
        empty_dir.mkdir()

        # Create directory containing ONLY hidden file
        hidden_only_dir = self.temp_dir_path / "class_hidden_only"
        hidden_only_dir.mkdir()
        (hidden_only_dir / ".gitkeep").touch()

        # Run validation
        report = self.validator.validate(self.temp_dir_path)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.total_files, 1)

        # Expected empty directories
        expected_empty = {empty_dir, hidden_only_dir}
        self.assertEqual(set(report.empty_dirs), expected_empty)

    def test_validate_detects_corrupted_images(self) -> None:
        """Tests that corrupted image files are identified."""
        ok_img = self.temp_dir_path / "ok.jpg"
        bad_img1 = self.temp_dir_path / "bad1.png"
        bad_img2 = self.temp_dir_path / "bad2.jpg"

        self.create_mock_image(ok_img, "JPEG")
        self.create_corrupted_file(bad_img1)
        self.create_corrupted_file(bad_img2)

        report = self.validator.validate(self.temp_dir_path)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.total_files, 3)  # Checked all 3 files
        self.assertEqual(set(report.invalid_files), {bad_img1, bad_img2})
        self.assertEqual(set(self.validator.list_invalid_files()), {bad_img1, bad_img2})

    def test_validate_archive_success(self) -> None:
        """Tests validate_archive returns True for correct ZIP archives."""
        zip_path = self.temp_dir_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("test.txt", "data")

        self.assertTrue(self.validator.validate_archive(zip_path))

    def test_validate_archive_nonexistent_or_dir(self) -> None:
        """Tests validate_archive returns False for invalid archive paths."""
        ghost_path = self.temp_dir_path / "ghost.zip"
        self.assertFalse(self.validator.validate_archive(ghost_path))

        dir_path = self.temp_dir_path / "folder.zip"
        dir_path.mkdir()
        self.assertFalse(self.validator.validate_archive(dir_path))

    def test_validate_archive_corrupted(self) -> None:
        """Tests validate_archive returns False for corrupted ZIP files."""
        zip_path = self.temp_dir_path / "corrupted.zip"
        with open(zip_path, "wb") as f:
            f.write(b"corrupted zip bytes")

        self.assertFalse(self.validator.validate_archive(zip_path))

    def test_validate_archive_invalid_type(self) -> None:
        """Tests that passing invalid type to validate_archive raises TypeError."""
        with self.assertRaises(TypeError):
            self.validator.validate_archive("string_path")  # type: ignore
