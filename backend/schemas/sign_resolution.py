"""Data contracts and models for Sign Resolution."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.schemas.sign_dictionary import SignEntry

class ResolutionStatus(str, Enum):
    """Status indicating whether a token/phrase was matched in the dictionary."""
    MATCHED = "MATCHED"
    UNSUPPORTED = "UNSUPPORTED"

class ResolvedToken(BaseModel):
    """Represents a segment of text and its resolution status."""
    token: str = Field(..., description="The original token or phrase text.")
    status: ResolutionStatus = Field(..., description="Whether the token was successfully matched.")
    sign: Optional[SignEntry] = Field(None, description="The matched dictionary sign entry, if any.")

class SignResolutionResult(BaseModel):
    """Result contract for a sign resolution operation."""
    resolved_tokens: List[ResolvedToken] = Field(..., description="List of resolved tokens preserving the original sequential order.")
