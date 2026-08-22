"""Test script for LandmarkDatasetBuilder module."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from backend.dataset.frame_sampler import FrameSampler
from backend.dataset.indexer import DatasetIndexer, DatasetSample
from backend.dataset.landmark_dataset_builder import LandmarkDatasetBuilder
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.normalizer import LandmarkNormalizer


def make_mock_sample(sample_id: str, class_name: str, class_index: int) -> DatasetSample:
    """Helper to create a synthetic DatasetSample."""
    return DatasetSample(
        sample_id=sample_id,
        class_name=class_name,
        class_index=class_index,
        video_path=Path(f"videos/{class_name}/{sample_id}"),
        frame_paths=tuple(Path(f"frame_{i:04d}.jpg") for i in range(1, 11)),
        num_frames=10,
    )


# Custom lightweight mocks for testing pipeline integration
class MockFrameSampler:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length

    def sample(self, paths: tuple[Path, ...]) -> list[Path]:
        return [Path(f"fake_frame_{i}.jpg") for i in range(32)]


class MockDetector:
    def detect(self, frame: np.ndarray) -> dict[str, any]:
        return {"success": True, "landmarks": [[[0.1, 0.2, 0.3]] * 21]}


class MockExtractor:
    def extract(self, result: dict[str, any]):
        from backend.sign_recognition.frame_features import FrameFeatures
        from backend.sign_recognition.hand_features import HandFeatures
        right = HandFeatures(np.ones(63, dtype=np.float32), "Right", 1.0)
        return FrameFeatures(right_hand=right)


class MockNormalizer:
    def normalize(self, features) -> np.ndarray:
        return np.ones(126, dtype=np.float32) * 5.0


def run_tests() -> None:
    """Runs verification tests for LandmarkDatasetBuilder."""
    results = {}
    mock_sample = make_mock_sample("mock/video", "mock_class", 0)

    # TEST 1 — Component initialization
    try:
        sampler = FrameSampler(sequence_length=32)
        detector = MediaPipeDetector()
        extractor = LandmarkExtractor()
        normalizer = LandmarkNormalizer()
        builder = LandmarkDatasetBuilder(sampler, detector, extractor, normalizer)

        assert builder.frame_sampler is sampler
        assert builder.detector is detector
        assert builder.extractor is extractor
        assert builder.normalizer is normalizer

        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # TEST 2 — Synthetic or mocked frame handling
    try:
        mock_sampler = MockFrameSampler(32)
        mock_detector = MockDetector()
        mock_extractor = MockExtractor()
        mock_normalizer = MockNormalizer()
        builder_mock = LandmarkDatasetBuilder(mock_sampler, mock_detector, mock_extractor, mock_normalizer)

        # Patch Path.exists to return True and cv2.imread to return a valid frame array
        with patch.object(Path, "exists", return_value=True), patch("cv2.imread") as mock_imread:
            mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            output = builder_mock.build_sample(mock_sample)

        assert output.shape == (32, 126), f"Expected shape (32, 126), got {output.shape}"
        assert np.all(output == 5.0), "Output vector values were not correctly preserved from mock normalization"
        results["Test 2: Synthetic frame handling"] = "PASSED"
    except Exception as e:
        results["Test 2: Synthetic frame handling"] = f"FAILED: {e}"

    # TEST 3 — Missing hand handling
    try:
        class MockMissingDetector:
            def detect(self, frame: np.ndarray) -> dict[str, any]:
                return {"success": False, "landmarks": []}

        real_extractor = LandmarkExtractor()
        real_normalizer = LandmarkNormalizer()
        builder_missing = LandmarkDatasetBuilder(
            mock_sampler, MockMissingDetector(), real_extractor, real_normalizer
        )

        with patch.object(Path, "exists", return_value=True), patch("cv2.imread") as mock_imread:
            mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            output = builder_missing.build_sample(mock_sample)

        assert output.shape == (32, 126), f"Expected shape (32, 126), got {output.shape}"
        # All 32 frames should have shape (126,) filled with exactly zeros
        assert np.all(output == 0.0), "Missing hands were not normalized to all zeros vector of size 126"
        results["Test 3: Missing hand handling"] = "PASSED"
    except Exception as e:
        results["Test 3: Missing hand handling"] = f"FAILED: {e}"

    # TEST 4 — Invalid frame path
    try:
        # Non-existent file path validation
        with patch.object(Path, "exists", return_value=False):
            try:
                builder_mock.build_sample(mock_sample)
                fnf_passed = False
            except FileNotFoundError:
                fnf_passed = True

        # cv2.imread returns None validation
        with patch.object(Path, "exists", return_value=True), patch("cv2.imread", return_value=None):
            try:
                builder_mock.build_sample(mock_sample)
                load_failed_passed = False
            except FileNotFoundError:
                load_failed_passed = True

        assert fnf_passed and load_failed_passed, "Error handling for non-existent or unloadable frames failed"
        results["Test 4: Invalid frame path"] = "PASSED"
    except Exception as e:
        results["Test 4: Invalid frame path"] = f"FAILED: {e}"

    # Print clean report of unit tests
    print("===================================")
    print("LANDMARK DATASET BUILDER TEST")
    print("===================================")
    print()
    for name, status in results.items():
        print(f"{name}")
        print(f"{status}")
        print()

    # TEST 5 — Real dataset integration
    dataset_root = Path("datasets/processed")
    real_report = None

    if dataset_root.exists() and dataset_root.is_dir():
        try:
            # 1. Initialize DatasetIndexer
            indexer = DatasetIndexer(dataset_root)
            # 2. Build the dataset index
            indexer.build_index()
            samples = indexer.get_samples()

            if samples:
                # 3. Select the first real sample
                real_sample = samples[0]

                # 4. Initialize builder with real project components
                real_sampler = FrameSampler(sequence_length=32)
                real_detector = MediaPipeDetector()
                real_extractor = LandmarkExtractor()
                real_normalizer = LandmarkNormalizer()
                real_builder = LandmarkDatasetBuilder(
                    real_sampler, real_detector, real_extractor, real_normalizer
                )

                # 5. Process that ONE sample only
                real_output = real_builder.build_sample(real_sample)

                # Counts frames with and without detected hands
                # A frame has no hands if the feature row is all zeros
                frames_without_hands = sum(1 for row in real_output if np.all(row == 0.0))
                frames_with_hands = 32 - frames_without_hands

                real_report = {
                    "sample_id": real_sample.sample_id,
                    "original_frames": real_sample.num_frames,
                    "output_shape": real_output.shape,
                    "feature_dim": real_output.shape[1],
                    "frames_with_hands": frames_with_hands,
                    "frames_without_hands": frames_without_hands,
                }
        except Exception as e:
            print(f"Error during real dataset integration test: {e}")

    if real_report:
        print("===================================")
        print("LANDMARK DATASET BUILDER REPORT")
        print("===================================")
        print()
        print(f"Sample ID: {real_report['sample_id']}")
        print()
        print(f"Original Frames: {real_report['original_frames']}")
        print(f"Sampled Frames: 32")
        print()
        print(f"Output Sequence Shape: {real_report['output_shape']}")
        print()
        print(f"Feature Dimension: {real_report['feature_dim']}")
        print()
        print(f"Frames With Detected Hands: {real_report['frames_with_hands']}")
        print(f"Frames Without Detected Hands: {real_report['frames_without_hands']}")
        print()

    any_failed = any("FAILED" in status for status in results.values())
    if any_failed:
        print("Some LandmarkDatasetBuilder tests FAILED!")
        sys.exit(1)
    else:
        print("All LandmarkDatasetBuilder tests completed successfully!")


if __name__ == "__main__":
    run_tests()
