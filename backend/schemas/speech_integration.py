"""Data contracts for speech and session integration."""

from pydantic import BaseModel
from backend.schemas.speech import SpeechToTextResult
from backend.schemas.speech_session import SpeechSession

class SessionTranscriptionResponse(BaseModel):
    """Integration response combining transcription result and session state."""
    session: SpeechSession
    transcription: SpeechToTextResult
