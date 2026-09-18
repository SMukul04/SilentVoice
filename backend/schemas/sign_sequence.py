"""Data contracts and models for ISL Sign Sequence."""

from typing import List
from pydantic import BaseModel, Field


class SignSequenceItem(BaseModel):
    """Represents a single ordered sign to be performed.
    
    This abstracts away the underlying dictionary and resolution process,
    providing just the identity and source metadata needed for Phase 7
    avatar playback mapping.
    """
    sign_id: str = Field(..., description="The unique identifier of the sign to perform.")
    source_text: str = Field(..., description="The original word or phrase that resolved to this sign.")
    original_index: int = Field(..., description="The 0-based position of this token in the resolved stream.")


class SignSequenceResult(BaseModel):
    """The constructed sequence and its metadata.
    
    Explicitly separates valid executable signs from unsupported content.
    """
    sequence: List[SignSequenceItem] = Field(
        ..., 
        description="The ordered list of signs to execute."
    )
    unsupported_tokens: List[str] = Field(
        ..., 
        description="Original text fragments that could not be resolved to signs."
    )
