"""Zenodo client for retrieving and parsing metadata of the INCLUDE dataset."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import requests

from backend.dataset.exceptions import ZenodoAPIError


@dataclass(slots=True)
class ZenodoFile:
    """Represents metadata for a single file hosted on Zenodo.

    Attributes:
        filename: Name of the file.
        size: Size of the file in bytes.
        checksum: Checksum hash of the file.
        download_url: Direct URL to download the file.
    """

    filename: str
    size: int
    checksum: str
    download_url: str


class ZenodoClient:
    """Communicates with the Zenodo REST API to retrieve and parse metadata."""

    BASE_URL: str = "https://zenodo.org/api/records"

    def __init__(self, record_id: int) -> None:
        """Initializes the ZenodoClient with a record ID.

        Parameters
        ----------
        record_id : int
            The Zenodo record identifier to query.
        """
        self.record_id: int = record_id
        self._metadata: dict[str, typing.Any] | None = None
        self._files: list[ZenodoFile] | None = None

    @property
    def metadata(self) -> dict[str, typing.Any] | None:
        """Returns the retrieved raw metadata.

        Returns:
            The raw metadata dictionary, or None if not fetched yet.
        """
        return self._metadata

    @property
    def files(self) -> list[ZenodoFile] | None:
        """Returns the list of files parsed from metadata.

        Returns:
            A list of ZenodoFile objects, or None if not fetched yet.
        """
        return self._files

    def fetch_metadata(self) -> None:
        """Retrieves metadata from the Zenodo REST API and parses files.

        Builds the request URL, sends a GET request, validates the response,
        and populates the internal metadata and files list.

        Raises
        ------
        ZenodoAPIError
            If the API request fails, times out, returns an invalid HTTP status,
            or contains invalid JSON.
        """
        url = f"{self.BASE_URL}/{self.record_id}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ZenodoAPIError(f"Zenodo API request failed: {e}") from e
        except ValueError as e:
            raise ZenodoAPIError(
                f"Failed to decode JSON from Zenodo API response: {e}"
            ) from e

        self._metadata = data
        self._parse_files(data)

    def _parse_files(self, data: dict[str, typing.Any]) -> None:
        """Parses files information from raw metadata JSON.

        Parameters
        ----------
        data : dict
            The raw Zenodo API JSON response dictionary.

        Raises
        ------
        ZenodoAPIError
            If files section is missing or has an invalid format.
        """
        files_data = data.get("files")
        if files_data is None:
            raise ZenodoAPIError("Invalid Zenodo response: 'files' key is missing.")

        if not isinstance(files_data, list):
            raise ZenodoAPIError("Invalid Zenodo response: 'files' must be a list.")

        parsed_files = []
        for file_info in files_data:
            try:
                filename = file_info.get("key") or file_info.get("filename")
                size = file_info.get("size")
                checksum = file_info.get("checksum")

                links = file_info.get("links", {})
                download_url = (
                    links.get("self") or links.get("download") or links.get("content")
                )

                if not filename or size is None or not checksum or not download_url:
                    raise ZenodoAPIError(
                        f"Missing required file metadata: filename={filename}, "
                        f"size={size}, checksum={checksum}, download_url={download_url}"
                    )

                parsed_files.append(
                    ZenodoFile(
                        filename=str(filename),
                        size=int(size),
                        checksum=str(checksum),
                        download_url=str(download_url),
                    )
                )
            except (TypeError, ValueError) as e:
                raise ZenodoAPIError(f"Failed to parse file metadata: {e}") from e

        self._files = parsed_files

    def get_title(self) -> str:
        """Returns the dataset title.

        Returns
        -------
        str
            The dataset title.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._metadata is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        metadata_dict = self._metadata.get("metadata", {})
        title = metadata_dict.get("title") or self._metadata.get("title")
        if not title:
            return ""
        return str(title)

    def get_version(self) -> str:
        """Returns the version string of the dataset.

        Returns
        -------
        str
            The version string, or an empty string if not available.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._metadata is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        metadata_dict = self._metadata.get("metadata", {})
        version = metadata_dict.get("version") or self._metadata.get("version")
        if version is None:
            return ""
        return str(version)

    def get_doi(self) -> str:
        """Returns the dataset DOI (Digital Object Identifier).

        Returns
        -------
        str
            The DOI string, or an empty string if not available.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._metadata is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        metadata_dict = self._metadata.get("metadata", {})
        doi = metadata_dict.get("doi") or self._metadata.get("doi")
        if doi is None:
            return ""
        return str(doi)

    def get_files(self) -> list[ZenodoFile]:
        """Returns a copy of the list of files parsed from metadata.

        Returns
        -------
        list of ZenodoFile
            A copy of the list of ZenodoFile objects.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._files is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        return list(self._files)

    def get_total_size(self) -> int:
        """Returns the total size of all files in bytes.

        Returns
        -------
        int
            The total size in bytes.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._files is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        return sum(f.size for f in self._files)

    def get_download_urls(self) -> list[str]:
        """Returns a list of direct download URLs for all files in the dataset.

        Returns
        -------
        list of str
            A list of download URL strings.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        """
        if self._files is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        return [f.download_url for f in self._files]

    def get_file(self, filename: str) -> ZenodoFile:
        """Returns the matching ZenodoFile for the given filename.

        Parameters
        ----------
        filename : str
            The name of the file to search for.

        Returns
        -------
        ZenodoFile
            The matching ZenodoFile instance.

        Raises
        ------
        ValueError
            If metadata has not been fetched yet.
        KeyError
            If the file does not exist in the dataset.
        """
        if self._files is None:
            raise ValueError("Metadata has not been fetched. Call fetch_metadata() first.")

        for f in self._files:
            if f.filename == filename:
                return f

        raise KeyError(f"File '{filename}' not found in dataset record.")
