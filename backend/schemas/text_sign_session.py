"""Data contracts for Text to Sign Session Management."""

from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

class TextSignSessionState(str, Enum):
    """Explicit lifecycle states for a text-to-sign session."""
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class TextSignSession(BaseModel):
    """Represents the state of an active text-to-sign session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: TextSignSessionState = Field(default=TextSignSessionState.IDLE)
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)

class SessionNotFoundError(Exception):
    """Raised when a session cannot be found."""
    pass

class InvalidStateTransitionError(Exception):
    """Raised when a state transition is not allowed."""
    pass
