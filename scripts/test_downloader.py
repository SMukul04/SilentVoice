from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.zenodo_client import ZenodoClient
from backend.dataset.downloader import DatasetDownloader


def main():
    # Initialize Zenodo client
    client = ZenodoClient(4010759)
    client.fetch_metadata()

    # Initialize downloader
    downloader = DatasetDownloader(
        client=client,
        download_directory=Path("datasets")
    )

    # Download ONLY the first archive
    first_file = client.get_files()[0]

    print(f"Downloading: {first_file.filename}")
    print(f"Size: {first_file.size / (1024**2):.2f} MB")

    downloader.download_file(first_file)

    print("Download completed successfully!")


if __name__ == "__main__":
    main()