"""Data contracts and models for the Sign Dictionary."""

from typing import Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Exceptions
# ==============================================================================

class SignDictionaryError(Exception):
    """Base exception for Sign Dictionary errors."""
    pass

class SignNotFoundError(SignDictionaryError):
    """Raised when a requested sign is not found in the dictionary."""
    pass

class DuplicateSignError(SignDictionaryError):
    """Raised during initialization if duplicate signs are detected."""
    pass


# ==============================================================================
# Models
# ==============================================================================

class SignEntry(BaseModel):
    """Represents a single ISL sign in the dictionary.
    
    This represents the vocabulary identity of a sign, completely decoupled
    from how it is recognized by ML or animated by an avatar.
    """
    sign_id: str = Field(..., description="The unique immutable identifier for this sign (e.g. 'HELLO').")
    canonical_name: str = Field(..., description="The standard English word/phrase for this sign (e.g. 'hello').")
    description: Optional[str] = Field(None, description="Optional description of the sign.")
