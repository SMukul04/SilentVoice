"""Test script for the FrameExtractor module to verify video frame extraction and caching."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to resolve imports correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.frame_extractor import FrameExtractor


def count_processed_folders(output_dir: Path) -> int:
    """Helper to count the number of directories containing at least one JPEG frame.

    Parameters
    ----------
    output_dir : Path
        The output directory to scan.

    Returns
    -------
    int
        The count of directories containing at least one .jpg file.
    """
    count = 0
    if not output_dir.exists():
        return count

    try:
        for p in output_dir.rglob("*"):
            if p.is_dir():
                try:
                    # Check if the folder contains any file ending in .jpg
                    if any(child.is_file() and child.suffix.lower() == ".jpg" for child in p.iterdir()):
                        count += 1
                except OSError:
                    pass
    except OSError:
        pass

    return count


def main() -> None:
    """Instantiates FrameExtractor and runs frame extraction tests."""
    extractor = FrameExtractor()

    dataset_root = Path("datasets/raw")
    output_root = Path("datasets/processed")

    print("Running initial frame extraction...")
    stats1 = extractor.extract_directory(dataset_root, output_root)

    print("\n===================================")
    print("Frame Extraction Report")
    print("===================================")
    print(f"Processed Videos : {stats1['processed']}")
    print(f"Skipped Videos   : {stats1['skipped']}")
    print(f"Failed Videos    : {stats1['failed']}")
    print("===================================\n")

    print("Running frame extraction a SECOND TIME automatically...")
    stats2 = extractor.extract_directory(dataset_root, output_root)

    print("\n===================================")
    print("Second Run")
    print("===================================")
    print(f"Processed : {stats2['processed']}")
    print(f"Skipped   : {stats2['skipped']}")
    print(f"Failed    : {stats2['failed']}")
    print("===================================\n")

    total_folders = count_processed_folders(output_root)
    print(f"Total processed video folders created: {total_folders}")


if __name__ == "__main__":
    main()
