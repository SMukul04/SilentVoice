"""Data contracts for Speech Session Management."""

from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

class SessionState(str, Enum):
    """Explicit lifecycle states for a speech session."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class SpeechSession(BaseModel):
    """Represents the state of an active speech session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = Field(default=SessionState.IDLE)
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)
