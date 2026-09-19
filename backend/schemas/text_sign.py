"""Data contracts for Text to Sign requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field
from backend.schemas.text_sign_session import TextSignSessionState
from backend.schemas.sign_sequence import SignSequenceItem

class TextSignRequest(BaseModel):
    """Request payload for processing text."""
    text: str = Field(..., description="The text to convert to a sign sequence.")

class TextSignResponse(BaseModel):
    """Combined response representing the outcome of text-to-sign processing."""
    session_id: str = Field(..., description="The ID of the session.")
    state: TextSignSessionState = Field(..., description="The current state of the session.")
    original_text: str = Field(..., description="The original unmodified text.")
    normalized_text: str = Field(..., description="The normalized text after processing.")
    sequence: List[SignSequenceItem] = Field(..., description="The resolved sign sequence.")
    unsupported_tokens: List[str] = Field(..., description="Tokens that could not be resolved.")
