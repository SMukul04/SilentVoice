"""Unit tests for the DatasetExtractor class."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.dataset.exceptions import ExtractionError
from backend.dataset.extractor import DatasetExtractor


class TestDatasetExtractor(unittest.TestCase):
    """Unit tests to verify functionality and safety of DatasetExtractor."""

    def setUp(self) -> None:
        """Sets up temporary directories for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.output_dir = self.temp_dir_path / "raw"
        self.extractor = DatasetExtractor(output_dir=self.output_dir)

    def tearDown(self) -> None:
        """Cleans up temporary directories."""
        self.temp_dir.cleanup()

    def test_init_invalid_type(self) -> None:
        """Tests that passing an invalid type to the constructor raises TypeError."""
        with self.assertRaises(TypeError):
            DatasetExtractor(output_dir="not/a/path")  # type: ignore

    def test_list_archives_success(self) -> None:
        """Tests that list_archives correctly lists and sorts ZIP files."""
        # Create dummy files
        zip1 = self.temp_dir_path / "archive_b.zip"
        zip2 = self.temp_dir_path / "archive_a.zip"
        txt_file = self.temp_dir_path / "readme.txt"

        zip1.touch()
        zip2.touch()
        txt_file.touch()

        archives = self.extractor.list_archives(self.temp_dir_path)
        # Should be sorted alphabetically and exclude txt
        self.assertEqual(archives, [zip2, zip1])

    def test_list_archives_invalid_directory(self) -> None:
        """Tests that list_archives raises appropriate errors for invalid inputs."""
        with self.assertRaises(TypeError):
            self.extractor.list_archives("invalid/type")  # type: ignore

        with self.assertRaises(FileNotFoundError):
            self.extractor.list_archives(self.temp_dir_path / "nonexistent_dir")

    def test_is_extracted(self) -> None:
        """Tests is_extracted behavior on missing/extracted archives."""
        zip_path = self.temp_dir_path / "test.zip"

        # 1. Non-existent archive -> False
        self.assertFalse(self.extractor.is_extracted(zip_path))

        # Create a valid ZIP
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("folder/file1.txt", "content1")
            z.writestr("file2.txt", "content2")

        # 2. Before extraction -> False
        self.assertFalse(self.extractor.is_extracted(zip_path))

        # 3. After partial/complete manual extraction simulation
        # Create file2.txt but not folder/file1.txt
        (self.output_dir / "file2.txt").touch()
        self.assertFalse(self.extractor.is_extracted(zip_path))

        # Create folder/file1.txt
        (self.output_dir / "folder").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "folder" / "file1.txt").touch()
        self.assertTrue(self.extractor.is_extracted(zip_path))

    def test_extract_success(self) -> None:
        """Tests successful ZIP extraction and return path."""
        zip_path = self.temp_dir_path / "test_extract.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("subdir/hello.txt", "hello world")
            z.writestr("root.txt", "root file")

        # Extract
        result_path = self.extractor.extract(zip_path)
        self.assertEqual(result_path, self.output_dir)

        # Verify files exist and have correct content
        file1 = self.output_dir / "subdir" / "hello.txt"
        file2 = self.output_dir / "root.txt"

        self.assertTrue(file1.exists())
        self.assertTrue(file2.exists())

        with open(file1, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello world")

    def test_extract_skip_if_already_extracted(self) -> None:
        """Tests that extract skips work if the zip has already been extracted."""
        zip_path = self.temp_dir_path / "test_skip.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("skip.txt", "skip content")

        # First extraction
        self.extractor.extract(zip_path)
        first_modified_time = (self.output_dir / "skip.txt").stat().st_mtime

        # Modify file manually so we can detect if it got overwritten
        with open(self.output_dir / "skip.txt", "w", encoding="utf-8") as f:
            f.write("manually modified")

        # Second extraction -> should be skipped
        self.extractor.extract(zip_path)

        with open(self.output_dir / "skip.txt", "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "manually modified")

    def test_extract_invalid_zip(self) -> None:
        """Tests that extract raises ExtractionError for invalid ZIP files."""
        zip_path = self.temp_dir_path / "invalid.zip"
        # Write corrupted content
        with open(zip_path, "w", encoding="utf-8") as f:
            f.write("not a zip file content")

        with self.assertRaises(ExtractionError):
            self.extractor.extract(zip_path)

    def test_extract_non_existent(self) -> None:
        """Tests that extract raises FileNotFoundError for non-existent archives."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract(self.temp_dir_path / "ghost.zip")

    def test_extract_invalid_types(self) -> None:
        """Tests that extract raises TypeError for invalid argument types."""
        with self.assertRaises(TypeError):
            self.extractor.extract("string_path")  # type: ignore

    def test_extract_path_traversal_detection(self) -> None:
        """Tests that ZIP archives attempting path traversal (Zip Slip) raise ExtractionError."""
        zip_path = self.temp_dir_path / "traversal.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("../traversal_escape.txt", "escape")

        with self.assertRaises(ExtractionError) as ctx:
            self.extractor.extract(zip_path)

        self.assertIn("Path traversal", str(ctx.exception))

    def test_extract_all_success(self) -> None:
        """Tests that extract_all extracts multiple archives sequentially."""
        # Create two archives in a dedicated downloads dir
        downloads_dir = self.temp_dir_path / "downloads"
        downloads_dir.mkdir()

        zip1 = downloads_dir / "archive1.zip"
        zip2 = downloads_dir / "archive2.zip"

        with zipfile.ZipFile(zip1, "w") as z:
            z.writestr("a1.txt", "archive 1 file")
        with zipfile.ZipFile(zip2, "w") as z:
            z.writestr("a2.txt", "archive 2 file")

        results = self.extractor.extract_all(downloads_dir)
        self.assertEqual(results, [self.output_dir, self.output_dir])

        self.assertTrue((self.output_dir / "a1.txt").exists())
        self.assertTrue((self.output_dir / "a2.txt").exists())

    def test_extract_all_invalid_types(self) -> None:
        """Tests that extract_all raises TypeError/FileNotFoundError for invalid paths."""
        with self.assertRaises(TypeError):
            self.extractor.extract_all("string_path")  # type: ignore

        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_all(self.temp_dir_path / "ghost_dir")
