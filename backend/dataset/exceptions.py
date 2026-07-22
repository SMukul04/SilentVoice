"""Custom exception hierarchy for the SilentVoice dataset package."""

from __future__ import annotations


class DatasetError(Exception):
    """Base exception for all dataset-related operations in SilentVoice."""


class MetadataError(DatasetError):
    """Raised when an error occurs while loading, validating, or parsing dataset metadata."""


class ZenodoAPIError(DatasetError):
    """Raised when communication with the Zenodo REST API fails or returns invalid metadata."""


class DownloadError(DatasetError):
    """Raised when download operations fail or are interrupted."""


class ExtractionError(DatasetError):
    """Raised when dataset archive extraction or processing fails."""


class ValidationError(DatasetError):
    """Raised when data integrity or schema validation checks fail."""
