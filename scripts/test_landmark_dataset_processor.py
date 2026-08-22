"""Test script for LandmarkDatasetProcessor module."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import numpy as np

from backend.dataset.frame_sampler import FrameSampler
from backend.dataset.indexer import DatasetIndexer, DatasetSample
from backend.dataset.splitter import DatasetSplit, DatasetSplitter
from backend.dataset.landmark_dataset_builder import LandmarkDatasetBuilder
from backend.dataset.landmark_dataset_processor import LandmarkDatasetProcessor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.normalizer import LandmarkNormalizer


# Custom lightweight mocks for testing pipeline integration
class MockIndexer:
    def __init__(self, samples: list[DatasetSample]) -> None:
        self.samples = samples

    def build_index(self) -> None:
        pass

    def get_num_samples(self) -> int:
        return len(self.samples)

    def get_samples(self) -> list[DatasetSample]:
        return self.samples

    def get_num_classes(self) -> int:
        return 2

    def get_class_to_index(self) -> dict[str, int]:
        return {"class_A": 0, "class_B": 1}

    def get_index_to_class(self) -> dict[int, str]:
        return {0: "class_A", 1: "class_B"}


class MockSplitter:
    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_train: list[DatasetSample] | None = None,
        split_val: list[DatasetSample] | None = None,
        split_test: list[DatasetSample] | None = None,
    ) -> None:
        self.random_seed = seed
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.split_train = split_train or []
        self.split_val = split_val or []
        self.split_test = split_test or []

    def split(self, samples: list[DatasetSample]) -> DatasetSplit:
        return DatasetSplit(
            train=self.split_train,
            validation=self.split_val,
            test=self.split_test,
        )


class MockSampler:
    def __init__(self, sequence_length: int = 32) -> None:
        self.sequence_length = sequence_length


class MockBuilder:
    def __init__(self, sequence_length: int = 32, return_shape: tuple[int, int] = (32, 126)) -> None:
        self.frame_sampler = MockSampler(sequence_length)
        self.return_shape = return_shape

    def build_sample(self, sample: DatasetSample) -> np.ndarray:
        if self.return_shape != (32, 126):
            return np.ones(self.return_shape, dtype=np.float32)
        # Return unique array filled with class_index + 1
        return np.ones((32, 126), dtype=np.float32) * float(sample.class_index + 1)


def run_tests() -> None:
    """Runs verification tests for LandmarkDatasetProcessor."""
    results = {}

    # TEST 1 — Component initialization
    try:
        tmp_dir = Path("dummy_output_dir")
        # Instantiate real dependencies
        real_sampler = FrameSampler(sequence_length=32)
        real_detector = MediaPipeDetector()
        real_extractor = LandmarkExtractor()
        real_normalizer = LandmarkNormalizer()
        real_builder = LandmarkDatasetBuilder(
            real_sampler, real_detector, real_extractor, real_normalizer
        )
        real_indexer = DatasetIndexer(Path("datasets/processed"))
        real_splitter = DatasetSplitter()

        processor = LandmarkDatasetProcessor(
            real_indexer, real_splitter, real_builder, tmp_dir
        )

        assert processor.indexer is real_indexer
        assert processor.splitter is real_splitter
        assert processor.builder is real_builder
        assert processor.output_dir is tmp_dir

        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # TEST 2 — Synthetic split processing
    try:
        # Create 3 synthetic samples
        samples = [
            DatasetSample("A1", "class_A", 0, Path("A1"), (), 0),
            DatasetSample("A2", "class_A", 0, Path("A2"), (), 0),
            DatasetSample("B1", "class_B", 1, Path("B1"), (), 0),
        ]
        mock_indexer = MockIndexer(samples)
        mock_splitter = MockSplitter(split_train=samples)
        mock_builder = MockBuilder()

        processor_mock = LandmarkDatasetProcessor(
            mock_indexer, mock_splitter, mock_builder, Path("tmp_out")
        )

        # Process training split
        X, y = processor_mock.process_split(samples, "train")

        # Verify X has expected shape (3, 32, 126)
        assert X.shape == (3, 32, 126), f"Expected shape (3, 32, 126), got {X.shape}"
        assert y.shape == (3,), f"Expected shape (3,), got {y.shape}"

        # Verify X uses float32 and y has integer dtype
        assert X.dtype == np.float32, f"Expected dtype float32, got {X.dtype}"
        assert np.issubdtype(y.dtype, np.integer), f"Expected integer dtype, got {y.dtype}"

        # Verify labels match expected class indices
        assert np.array_equal(y, [0, 0, 1]), f"Expected labels [0, 0, 1], got {y}"

        # Verify sequence ordering is correct
        assert np.all(X[0] == 1.0), "X[0] values do not match class_index 0 mock payload"
        assert np.all(X[1] == 1.0), "X[1] values do not match class_index 0 mock payload"
        assert np.all(X[2] == 2.0), "X[2] values do not match class_index 1 mock payload"

        results["Test 2: Synthetic split processing"] = "PASSED"
    except Exception as e:
        results["Test 2: Synthetic split processing"] = f"FAILED: {e}"

    # TEST 3 — Inconsistent sequence shape
    try:
        samples = [DatasetSample("A1", "class_A", 0, Path("A1"), (), 0)]
        mock_indexer = MockIndexer(samples)
        mock_splitter = MockSplitter(split_train=samples)

        # Test case A: sequence length is 31 instead of 32
        mock_builder_31 = MockBuilder(return_shape=(31, 126))
        processor_31 = LandmarkDatasetProcessor(
            mock_indexer, mock_splitter, mock_builder_31, Path("tmp_out")
        )
        try:
            processor_31.process_split(samples, "train")
            raised_31 = False
        except ValueError as e:
            raised_31 = True
            assert "32" in str(e) and "126" in str(e), f"Error message '{e}' did not mention expected shape"
        assert raised_31, "Should raise ValueError for incorrect sequence length"

        # Test case B: feature dimension is 125 instead of 126
        mock_builder_125 = MockBuilder(return_shape=(32, 125))
        processor_125 = LandmarkDatasetProcessor(
            mock_indexer, mock_splitter, mock_builder_125, Path("tmp_out")
        )
        try:
            processor_125.process_split(samples, "train")
            raised_125 = False
        except ValueError as e:
            raised_125 = True
            assert "32" in str(e) and "126" in str(e), f"Error message '{e}' did not mention expected shape"
        assert raised_125, "Should raise ValueError for incorrect feature dimension"

        results["Test 3: Inconsistent sequence shape"] = "PASSED"
    except Exception as e:
        results["Test 3: Inconsistent sequence shape"] = f"FAILED: {e}"

    # TEST 4 — Saving and loading
    try:
        X_orig = np.random.randn(3, 32, 126).astype(np.float32)
        y_orig = np.array([0, 1, 0], dtype=np.int64)

        mock_indexer = MockIndexer([])
        mock_splitter = MockSplitter()
        mock_builder = MockBuilder()

        with tempfile.TemporaryDirectory(dir=".") as tmp_test_dir:
            tmp_path = Path(tmp_test_dir)
            proc_tmp = LandmarkDatasetProcessor(
                mock_indexer, mock_splitter, mock_builder, tmp_path
            )

            # Save split
            proc_tmp.save_split(X_orig, y_orig, "train.npz")

            # Load and verify they match exactly
            file_path = tmp_path / "train.npz"
            with np.load(file_path) as loaded:
                assert "X" in loaded and "y" in loaded, "NPZ file does not contain X and y arrays"
                X_loaded = loaded["X"].copy()
                y_loaded = loaded["y"].copy()

            assert np.array_equal(X_loaded, X_orig), "Loaded X features do not match original"
            assert np.array_equal(y_loaded, y_orig), "Loaded y labels do not match original"

        results["Test 4: Saving and loading"] = "PASSED"
    except Exception as e:
        results["Test 4: Saving and loading"] = f"FAILED: {e}"

    # TEST 5 — Metadata generation
    try:
        mock_indexer = MockIndexer([])
        mock_splitter = MockSplitter(seed=42, train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15)
        mock_builder = MockBuilder()

        mock_split = DatasetSplit(
            train=[
                DatasetSample("A1", "class_A", 0, Path("A1"), (), 0),
                DatasetSample("A2", "class_A", 0, Path("A2"), (), 0),
            ],
            validation=[
                DatasetSample("B1", "class_B", 1, Path("B1"), (), 0),
            ],
            test=[
                DatasetSample("B2", "class_B", 1, Path("B2"), (), 0),
            ]
        )

        with tempfile.TemporaryDirectory(dir=".") as tmp_test_dir:
            tmp_path = Path(tmp_test_dir)
            proc_tmp = LandmarkDatasetProcessor(
                mock_indexer, mock_splitter, mock_builder, tmp_path
            )
            proc_tmp.save_metadata(mock_split)

            # Load and verify JSON contents
            with open(tmp_path / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            assert meta["num_classes"] == 2
            assert meta["class_to_index"] == {"class_A": 0, "class_B": 1}
            assert meta["index_to_class"] == {"0": "class_A", "1": "class_B"}
            assert meta["sequence_length"] == 32
            assert meta["feature_dimension"] == 126
            assert meta["train_sample_count"] == 2
            assert meta["validation_sample_count"] == 1
            assert meta["test_sample_count"] == 1
            assert meta["random_seed"] == 42
            assert meta["split_ratios"] == {"train": 0.70, "validation": 0.15, "test": 0.15}

        results["Test 5: Metadata generation"] = "PASSED"
    except Exception as e:
        results["Test 5: Metadata generation"] = f"FAILED: {e}"

    # Print unit test results
    print("===================================")
    print("LANDMARK DATASET PROCESSOR TEST")
    print("===================================")
    print()
    for name, status in results.items():
        print(f"{name}")
        print(f"{status}")
        print()

    # TEST 6 — REAL DATASET INTEGRATION
    real_dataset_root = Path("datasets/processed")
    real_output_dir = Path("datasets/landmarks")
    real_report = None

    if real_dataset_root.exists() and real_dataset_root.is_dir():
        try:
            # Re-initialize real components
            real_sampler = FrameSampler(sequence_length=32)
            real_detector = MediaPipeDetector()
            real_extractor = LandmarkExtractor()
            real_normalizer = LandmarkNormalizer()
            real_builder = LandmarkDatasetBuilder(
                real_sampler, real_detector, real_extractor, real_normalizer
            )
            real_indexer = DatasetIndexer(real_dataset_root)
            real_splitter = DatasetSplitter(
                train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15, random_seed=42
            )

            real_processor = LandmarkDatasetProcessor(
                real_indexer, real_splitter, real_builder, real_output_dir
            )

            # Run COMPLETE pipeline
            real_processor.process()

            # Verify saved files exist
            train_npz = real_output_dir / "train.npz"
            val_npz = real_output_dir / "validation.npz"
            test_npz = real_output_dir / "test.npz"
            metadata_json = real_output_dir / "metadata.json"

            assert train_npz.exists(), f"Missing file: {train_npz}"
            assert val_npz.exists(), f"Missing file: {val_npz}"
            assert test_npz.exists(), f"Missing file: {test_npz}"
            assert metadata_json.exists(), f"Missing file: {metadata_json}"

            # Load to verify shapes
            train_data = np.load(train_npz)
            val_data = np.load(val_npz)
            test_data = np.load(test_npz)

            real_report = {
                "num_classes": real_indexer.get_num_classes(),
                "train_X_shape": train_data["X"].shape,
                "train_y_shape": train_data["y"].shape,
                "val_X_shape": val_data["X"].shape,
                "val_y_shape": val_data["y"].shape,
                "test_X_shape": test_data["X"].shape,
                "test_y_shape": test_data["y"].shape,
                "saved_files": [
                    str(train_npz),
                    str(val_npz),
                    str(test_npz),
                    str(metadata_json),
                ],
            }
            results["Test 6: Real dataset integration"] = "PASSED"
        except Exception as e:
            results["Test 6: Real dataset integration"] = f"FAILED: {e}"

    if real_report:
        print("===================================")
        print("LANDMARK DATASET PROCESSING REPORT")
        print("===================================")
        print()
        print(f"Classes: {real_report['num_classes']}")
        print()
        print("Training:")
        print(f"X Shape: {real_report['train_X_shape']}")
        print(f"y Shape: {real_report['train_y_shape']}")
        print()
        print("Validation:")
        print(f"X Shape: {real_report['val_X_shape']}")
        print(f"y Shape: {real_report['val_y_shape']}")
        print()
        print("Test:")
        print(f"X Shape: {real_report['test_X_shape']}")
        print(f"y Shape: {real_report['test_y_shape']}")
        print()
        print("Sequence Length: 32")
        print("Feature Dimension: 126")
        print()
        print("Saved Files:")
        for path in real_report["saved_files"]:
            print(f"- {path}")
        print()

    any_failed = any(status.startswith("FAILED") for status in results.values())
    if any_failed:
        print("Some LandmarkDatasetProcessor tests FAILED!")
        sys.exit(1)
    else:
        print("All LandmarkDatasetProcessor tests completed successfully!")


if __name__ == "__main__":
    run_tests()
