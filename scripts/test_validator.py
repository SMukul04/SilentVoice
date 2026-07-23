from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.validator import DatasetValidator


def main() -> None:
    validator = DatasetValidator()

    dataset_root = Path("datasets/raw")

    report = validator.validate(dataset_root)

    print("\nValidation Report")
    print("-----------------")
    print(f"Valid: {report.is_valid}")
    print(f"Total Images: {report.total_files}")
    print(f"Invalid Files: {len(report.invalid_files)}")
    print(f"Empty Directories: {len(report.empty_dirs)}")

    if report.invalid_files:
        print("\nInvalid Files:")
        for file in report.invalid_files:
            print(f" - {file}")

    if report.empty_dirs:
        print("\nEmpty Directories:")
        for directory in report.empty_dirs:
            print(f" - {directory}")


if __name__ == "__main__":
    main()