"""Dataset downloader for fetching and managing SilentVoice dataset archives."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from backend.dataset.exceptions import DownloadError
from backend.dataset.zenodo_client import ZenodoClient, ZenodoFile


class DatasetDownloader:
    """Manages the downloading of dataset archives from Zenodo.

    Responsible only for downloading, verifying, and managing the local cache
    of the raw dataset archives.
    """

    def __init__(self, client: ZenodoClient, download_directory: Path) -> None:
        """Initializes the DatasetDownloader and creates the download directory.

        Parameters
        ----------
        client : ZenodoClient
            The Zenodo client used to inspect and verify dataset files.
        download_directory : Path
            The local directory path where files should be downloaded.
        """
        self.client = client
        self.download_directory = download_directory

        # Automatically create the directory if it does not exist
        self.download_directory.mkdir(parents=True, exist_ok=True)

    def download_file(self, file: ZenodoFile) -> None:
        """Downloads a single file from the Zenodo dataset.

        Parameters
        ----------
        file : ZenodoFile
            The file metadata object representing the file to download.

        Raises
        ------
        FileExistsError
            If the file already exists in the download directory.
        DownloadError
            If downloading fails or network error occurs.
        """
        dest_path = self.download_directory / file.filename
        if dest_path.exists():
            raise FileExistsError(f"File already exists at: {dest_path}")

        try:
            with requests.get(file.download_url, stream=True, timeout=30) as response:
                response.raise_for_status()

                content_length_str = response.headers.get("content-length")
                total_size = None
                if content_length_str is not None:
                    try:
                        total_size = int(content_length_str)
                    except ValueError:
                        pass

                with tqdm(
                    total=total_size,
                    desc=file.filename,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=True,
                ) as pbar:
                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
        except requests.RequestException as e:
            # Clean up the partially downloaded file if it was created
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            raise DownloadError(f"Failed to download file {file.filename}: {e}") from e

    def download_all(self) -> None:
        """Downloads all files associated with the dataset record.

        Raises
        ------
        DownloadError
            If any file fails to download.
        """
        files = self.client.get_files()
        for file in files:
            try:
                self.download_file(file)
            except FileExistsError:
                # Do not download again if file already exists
                pass

    def verify_checksum(self, file: ZenodoFile) -> bool:
        """Verifies the MD5 checksum of a downloaded file.

        Parameters
        ----------
        file : ZenodoFile
            The file metadata object to verify.

        Returns
        -------
        bool
            True if the checksum is valid, False otherwise.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("verify_checksum is not yet implemented.")

    def resume_download(self, file: ZenodoFile) -> None:
        """Resumes an interrupted file download.

        Parameters
        ----------
        file : ZenodoFile
            The file metadata object representing the file to resume.

        Raises
        ------
        DownloadError
            If resuming the download fails.
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("resume_download is not yet implemented.")

    def list_downloaded_files(self) -> list[Path]:
        """Lists the paths of all currently downloaded files.

        Returns
        -------
        list of Path
            A list of local file paths.
        """
        return [p for p in self.download_directory.iterdir() if p.is_file()]

    def get_download_directory(self) -> Path:
        """Returns the download directory path.

        Returns
        -------
        Path
            The path of the download directory.
        """
        return self.download_directory
