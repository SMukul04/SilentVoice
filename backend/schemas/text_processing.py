"""Data contracts and models for text processing operations."""

from pydantic import BaseModel, Field


# ==============================================================================
# Exceptions
# ==============================================================================

class TextProcessingError(Exception):
    """Base exception for all text processing errors."""
    pass

class EmptyTextError(TextProcessingError):
    """Raised when the input text is empty or contains only whitespace after normalization."""
    pass


# ==============================================================================
# Models
# ==============================================================================

class TextProcessingResult(BaseModel):
    """Result contract for a text processing operation."""
    original_text: str = Field(..., description="The original unmodified input text.")
    normalized_text: str = Field(..., description="The normalized text (lowercase, trimmed, collapsed whitespace).")
    tokens: list[str] = Field(..., description="The extracted tokens (words stripped of surrounding punctuation).")
