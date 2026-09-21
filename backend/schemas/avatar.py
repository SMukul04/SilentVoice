"""Data contracts and models for the 3D Avatar integration."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AvatarSignCommand(BaseModel):
    """Represents a request to the avatar to perform a specific sign."""
    sign_id: str = Field(..., min_length=1, description="The core semantic identity of the sign.")
    source_text: Optional[str] = Field(None, description="The original text word/phrase, if applicable.")
    sequence_index: int = Field(..., description="The order index in the current playback sequence.")


class AvatarSequence(BaseModel):
    """Represents an ordered sequence of signs for the avatar to perform."""
    sequence: List[AvatarSignCommand] = Field(
        ...,
        description="The ordered list of signs. Order must be strictly preserved."
    )
