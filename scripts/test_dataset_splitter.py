"""Test script for the DatasetSplitter module."""

from __future__ import annotations

from pathlib import Path
import sys

from backend.dataset.indexer import DatasetSample
from backend.dataset.splitter import DatasetSplitter


def make_mock_sample(sample_id: str, class_name: str, class_index: int) -> DatasetSample:
    """Helper to create a synthetic DatasetSample."""
    return DatasetSample(
        sample_id=sample_id,
        class_name=class_name,
        class_index=class_index,
        video_path=Path(f"videos/{class_name}/{sample_id}"),
        frame_paths=(),
        num_frames=0,
    )


def run_tests() -> None:
    """Runs verification tests for the DatasetSplitter."""
    results = {}

    # Setup synthetic samples
    # 8 samples per class for 3 classes (A, B, C)
    synthetic_samples = []
    for class_idx, class_name in enumerate(["class_A", "class_B", "class_C"]):
        for sample_idx in range(1, 9):
            sample_id = f"{class_name}/sample_{sample_idx}"
            synthetic_samples.append(make_mock_sample(sample_id, class_name, class_idx))

    # TEST 1 — Basic split
    try:
        splitter = DatasetSplitter(train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15, random_seed=42)
        split_result = splitter.split(synthetic_samples)

        # Check total output count equals input count
        assert split_result.total_count == len(synthetic_samples), (
            f"Total split count {split_result.total_count} != input count {len(synthetic_samples)}"
        )

        # Check that no duplicate samples exist across splits and every sample appears exactly once
        train_ids = {s.sample_id for s in split_result.train}
        val_ids = {s.sample_id for s in split_result.validation}
        test_ids = {s.sample_id for s in split_result.test}

        # Check overlap
        assert train_ids.isdisjoint(val_ids), "Train and validation sets overlap"
        assert train_ids.isdisjoint(test_ids), "Train and test sets overlap"
        assert val_ids.isdisjoint(test_ids), "Validation and test sets overlap"

        # Check that the union equals the input sample IDs
        all_ids = {s.sample_id for s in synthetic_samples}
        assert train_ids | val_ids | test_ids == all_ids, "Some samples are missing in splits"

        results["Test 1: Basic split"] = "PASSED"
    except Exception as e:
        results["Test 1: Basic split"] = f"FAILED: {e}"

    # TEST 2 — Class-aware distribution
    try:
        # We have 8 samples per class. Under 0.70 / 0.15 / 0.15 ratios:
        # Train should have 6, Validation 1, Test 1.
        for class_name in ["class_A", "class_B", "class_C"]:
            class_train = [s for s in split_result.train if s.class_name == class_name]
            class_val = [s for s in split_result.validation if s.class_name == class_name]
            class_test = [s for s in split_result.test if s.class_name == class_name]

            assert len(class_train) == 6, f"{class_name} train count is {len(class_train)} != 6"
            assert len(class_val) == 1, f"{class_name} validation count is {len(class_val)} != 1"
            assert len(class_test) == 1, f"{class_name} test count is {len(class_test)} != 1"

        results["Test 2: Class-aware distribution"] = "PASSED"
    except Exception as e:
        results["Test 2: Class-aware distribution"] = f"FAILED: {e}"

    # TEST 3 — Determinism
    try:
        splitter_1 = DatasetSplitter(random_seed=42)
        splitter_2 = DatasetSplitter(random_seed=42)

        split_1 = splitter_1.split(synthetic_samples)
        split_2 = splitter_2.split(synthetic_samples)

        assert [s.sample_id for s in split_1.train] == [s.sample_id for s in split_2.train], "Train sets differ"
        assert [s.sample_id for s in split_1.validation] == [s.sample_id for s in split_2.validation], "Val sets differ"
        assert [s.sample_id for s in split_1.test] == [s.sample_id for s in split_2.test], "Test sets differ"

        results["Test 3: Determinism"] = "PASSED"
    except Exception as e:
        results["Test 3: Determinism"] = f"FAILED: {e}"

    # TEST 4 — Different seed
    try:
        splitter_diff = DatasetSplitter(random_seed=100)
        split_diff = splitter_diff.split(synthetic_samples)

        # Split must remain valid
        assert split_diff.total_count == len(synthetic_samples)
        train_diff_ids = {s.sample_id for s in split_diff.train}
        val_diff_ids = {s.sample_id for s in split_diff.validation}
        test_diff_ids = {s.sample_id for s in split_diff.test}
        assert train_diff_ids.isdisjoint(val_diff_ids)
        assert train_diff_ids.isdisjoint(test_diff_ids)
        assert val_diff_ids.isdisjoint(test_diff_ids)

        results["Test 4: Different seed"] = "PASSED"
    except Exception as e:
        results["Test 4: Different seed"] = f"FAILED: {e}"

    # TEST 5 — Invalid ratios
    try:
        # Ratios do not sum to 1
        try:
            DatasetSplitter(train_ratio=0.5, validation_ratio=0.2, test_ratio=0.2)
            ratio_sum_ok = False
        except ValueError:
            ratio_sum_ok = True

        # Zero ratio
        try:
            DatasetSplitter(train_ratio=0.8, validation_ratio=0.2, test_ratio=0.0)
            zero_ok = False
        except ValueError:
            zero_ok = True

        # Negative ratio
        try:
            DatasetSplitter(train_ratio=0.9, validation_ratio=0.2, test_ratio=-0.1)
            neg_ok = False
        except ValueError:
            neg_ok = True

        # Non-numeric ratio
        try:
            DatasetSplitter(train_ratio="0.7", validation_ratio=0.15, test_ratio=0.15)  # type: ignore
            str_ok = False
        except TypeError:
            str_ok = True

        try:
            DatasetSplitter(train_ratio=True, validation_ratio=0.15, test_ratio=0.15)  # type: ignore
            bool_ratio_ok = False
        except TypeError:
            bool_ratio_ok = True

        assert ratio_sum_ok and zero_ok and neg_ok and str_ok and bool_ratio_ok, "Invalid ratio validation failed"
        results["Test 5: Invalid ratios"] = "PASSED"
    except Exception as e:
        results["Test 5: Invalid ratios"] = f"FAILED: {e}"

    # TEST 6 — Invalid random seed
    try:
        # float seed
        try:
            DatasetSplitter(random_seed=42.5)  # type: ignore
            float_seed_ok = False
        except TypeError:
            float_seed_ok = True

        # boolean seed
        try:
            DatasetSplitter(random_seed=True)  # type: ignore
            bool_seed_ok = False
        except TypeError:
            bool_seed_ok = True

        assert float_seed_ok and bool_seed_ok, "Invalid seed validation failed"
        results["Test 6: Invalid random seed"] = "PASSED"
    except Exception as e:
        results["Test 6: Invalid random seed"] = f"FAILED: {e}"

    # TEST 7 — Empty input
    try:
        splitter.split([])
        results["Test 7: Empty input"] = "FAILED: Did not raise exception on empty list"
    except ValueError:
        results["Test 7: Empty input"] = "PASSED"
    except Exception as e:
        results["Test 7: Empty input"] = f"FAILED: Raised unexpected exception: {e}"

    # Print clean report of unit tests
    print("===================================")
    print("DATASET SPLITTER TEST")
    print("===================================")
    print()
    for name, status in results.items():
        print(f"{name}")
        print(f"{status}")
        print()

    # REAL DATASET INTEGRATION
    dataset_root = Path("datasets/processed")
    real_split_result = None
    total_samples = 0

    if dataset_root.exists() and dataset_root.is_dir():
        try:
            from backend.dataset.indexer import DatasetIndexer
            indexer = DatasetIndexer(dataset_root)
            indexer.build_index()

            samples = indexer.get_samples()
            total_samples = len(samples)
            if samples:
                # Perform the split
                real_split_result = splitter.split(samples)

                # Verify:
                # 1. Total samples across all splits equals the indexed sample count.
                assert real_split_result.total_count == total_samples, "Total counts do not match"
                # 2. No sample appears in more than one split.
                real_train_ids = {s.sample_id for s in real_split_result.train}
                real_val_ids = {s.sample_id for s in real_split_result.validation}
                real_test_ids = {s.sample_id for s in real_split_result.test}
                assert real_train_ids.isdisjoint(real_val_ids), "Real split overlaps"
                assert real_train_ids.isdisjoint(real_test_ids), "Real split overlaps"
                assert real_val_ids.isdisjoint(real_test_ids), "Real split overlaps"
                # 3. The same seed produces the same result.
                real_split_result_second = splitter.split(samples)
                assert [s.sample_id for s in real_split_result.train] == [
                    s.sample_id for s in real_split_result_second.train
                ], "Real split determinism failed"
        except Exception as e:
            print(f"Error during real dataset integration: {e}")

    if real_split_result:
        print("===================================")
        print("DATASET SPLIT REPORT")
        print("===================================")
        print()
        print(f"Total Samples: {total_samples}")
        print()
        print(f"Training Samples: {real_split_result.train_count}")
        print(f"Validation Samples: {real_split_result.validation_count}")
        print(f"Test Samples: {real_split_result.test_count}")
        print()
        print("Class Distribution:")
        print()

        # Group split components by class
        all_classes = sorted({s.class_name for s in real_split_result.train} |
                             {s.class_name for s in real_split_result.validation} |
                             {s.class_name for s in real_split_result.test})

        for class_name in all_classes:
            tr_cnt = sum(1 for s in real_split_result.train if s.class_name == class_name)
            va_cnt = sum(1 for s in real_split_result.validation if s.class_name == class_name)
            te_cnt = sum(1 for s in real_split_result.test if s.class_name == class_name)
            print(class_name)
            print(f"  Train: {tr_cnt}")
            print(f"  Validation: {va_cnt}")
            print(f"  Test: {te_cnt}")
            print()

    any_failed = any("FAILED" in status for status in results.values())
    if any_failed:
        print("Some DatasetSplitter tests FAILED!")
        sys.exit(1)
    else:
        print("All DatasetSplitter tests completed successfully!")


if __name__ == "__main__":
    run_tests()
