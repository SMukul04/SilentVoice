"""Integration test script to verify ZenodoClient communication and parsing."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.zenodo_client import ZenodoClient


def main() -> None:
    """Main execution function for verifying the ZenodoClient integration."""
    print("Initializing ZenodoClient for record 4010759...")
    client = ZenodoClient(4010759)

    try:
        print("Fetching metadata from Zenodo REST API...")
        client.fetch_metadata()
        print("Metadata fetched successfully.\n")

        # 1. Dataset Title
        print("-" * 40)
        print("Dataset Title")
        print("-" * 40)
        print(client.get_title())
        print()

        # 2. Version
        print("-" * 40)
        print("Version")
        print("-" * 40)
        print(client.get_version() or "No version provided")
        print()

        # 3. DOI
        print("-" * 40)
        print("DOI")
        print("-" * 40)
        print(client.get_doi())
        print()

        # 4. Number of Files
        print("-" * 40)
        print("Number of Files")
        print("-" * 40)
        files = client.get_files()
        print(len(files))
        print()

        # 5. Total Dataset Size (GB)
        print("-" * 40)
        print("Total Dataset Size (GB)")
        print("-" * 40)
        total_size_bytes = client.get_total_size()
        total_size_gb = total_size_bytes / (1024 ** 3)
        print(f"{total_size_gb:.2f} GB")
        print()

        # 6. First Five Files
        print("-" * 40)
        print("First Five Files")
        print("-" * 40)
        first_five = files[:5]
        for f in first_five:
            size_mb = f.size / (1024 ** 2)
            print(f"Filename: {f.filename}")
            print(f"Size: {size_mb:.2f} MB")
            print(f"Checksum: {f.checksum}")
            print()

    except Exception as e:
        print(f"Integration test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
