"""Test script for the DatasetIndexer module."""

from __future__ import annotations

from pathlib import Path
import tempfile

from backend.dataset.indexer import DatasetIndexer


def run_tests() -> None:
    """Runs verification tests and prints a report on the processed dataset."""
    dataset_root = Path("datasets/processed")

    print(f"Initializing DatasetIndexer with root: {dataset_root}")
    indexer = DatasetIndexer(dataset_root)

    print("Building dataset index...")
    indexer.build_index()

    # Get stats
    num_classes = indexer.get_num_classes()
    num_samples = indexer.get_num_samples()
    total_frames = indexer.get_total_frames()

    # 4. Print a clear report
    print("\n===================================")
    print("Dataset Index Report")
    print("===================================")
    print(f"Classes: {num_classes}")
    print(f"Samples: {num_samples}")
    print(f"Total Frames: {total_frames}")
    print("===================================\n")

    # 5. Print the class mapping in deterministic order
    print("Class Mapping")
    print("-------------")
    class_to_index = indexer.get_class_to_index()
    # Sort by class index
    sorted_mappings = sorted(class_to_index.items(), key=lambda item: item[1])
    for class_name, class_idx in sorted_mappings:
        print(f"{class_idx} -> {class_name}")
    print()

    # 6. Print details for the first 5 samples, if available
    samples = indexer.get_samples()
    print("First 5 Sample Details")
    print("----------------------")
    for idx, sample in enumerate(samples[:5]):
        print(f"--- Sample {idx + 1} ---")
        print(f"Sample ID: {sample.sample_id}")
        print(f"Class: {sample.class_name}")
        print(f"Class Index: {sample.class_index}")
        print(f"Video Path: {sample.video_path}")
        print(f"Number of Frames: {sample.num_frames}")
        print(f"First Frame: {sample.frame_paths[0] if sample.frame_paths else 'N/A'}")
        print(f"Last Frame: {sample.frame_paths[-1] if sample.frame_paths else 'N/A'}")
    print()

    # 7. Test rebuild safety by calling build_index() a second time
    print("Testing rebuild safety...")
    indexer.build_index()

    # 8. Use assertions for the rebuild checks
    assert indexer.get_num_classes() == num_classes, (
        f"Class count changed after rebuild: {indexer.get_num_classes()} vs {num_classes}"
    )
    assert indexer.get_num_samples() == num_samples, (
        f"Sample count changed after rebuild: {indexer.get_num_samples()} vs {num_samples}"
    )
    assert indexer.get_total_frames() == total_frames, (
        f"Total frames count changed after rebuild: {indexer.get_total_frames()} vs {total_frames}"
    )
    print("Rebuild safety test passed successfully (counts remained identical).\n")

    # 9. Handle an empty dataset gracefully
    print("Testing empty dataset handling...")
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_indexer = DatasetIndexer(tmp_path)
        empty_indexer.build_index()

        assert empty_indexer.get_num_classes() == 0, "Empty dataset should have 0 classes"
        assert empty_indexer.get_num_samples() == 0, "Empty dataset should have 0 samples"
        assert empty_indexer.get_total_frames() == 0, "Empty dataset should have 0 total frames"
        print("Empty dataset test passed successfully!\n")

    print("All tests completed successfully!")


if __name__ == "__main__":
    run_tests()
