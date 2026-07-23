from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.extractor import DatasetExtractor


def main() -> None:
    extractor = DatasetExtractor()

    zip_path = Path("datasets/Adjectives_3of8.zip")

    print(f"Extracting: {zip_path.name}")

    output_dir = extractor.extract(zip_path)

    print(f"Extracted to: {output_dir}")


if __name__ == "__main__":
    main()