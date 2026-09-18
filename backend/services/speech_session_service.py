"""Speech session management service.

Provides a provider-independent speech session lifecycle layer that coordinates
speech operations and states without coupling to specific STT/TTS implementations.
"""

import threading
from typing import Dict
from datetime import datetime, timezone
import uuid

from backend.schemas.speech_session import SpeechSession, SessionState
from backend.schemas.speech import SessionNotFoundError, InvalidStateTransitionError

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class SpeechSessionService:
    """Manages speech session lifecycles, state transitions, and isolation."""

    def __init__(self):
        self._sessions: Dict[str, SpeechSession] = {}
        self._lock = threading.Lock()
        
        # Define allowed state transitions
        self.valid_transitions = {
            SessionState.IDLE: {
                SessionState.LISTENING, 
                SessionState.TRANSCRIBING, 
                SessionState.ERROR, 
                SessionState.CLOSED
            },
            SessionState.LISTENING: {
                SessionState.TRANSCRIBING, 
                SessionState.ERROR, 
                SessionState.CLOSED
            },
            SessionState.TRANSCRIBING: {
                SessionState.COMPLETED, 
                SessionState.ERROR, 
                SessionState.CLOSED
            },
            SessionState.SYNTHESIZING: {
                SessionState.COMPLETED, 
                SessionState.ERROR, 
                SessionState.CLOSED
            },
            SessionState.COMPLETED: {
                SessionState.SYNTHESIZING, 
                SessionState.IDLE, 
                SessionState.CLOSED
            },
            SessionState.ERROR: {
                SessionState.IDLE,
                SessionState.CLOSED
            },
            SessionState.CLOSED: set()  # Terminal state
        }

    def create_session(self) -> SpeechSession:
        """Create a new speech session with a unique ID."""
        with self._lock:
            session_id = str(uuid.uuid4())
            session = SpeechSession(
                session_id=session_id,
                state=SessionState.IDLE,
                created_at=_now_utc(),
                updated_at=_now_utc()
            )
            self._sessions[session_id] = session
            return session.model_copy()

    def get_session(self, session_id: str) -> SpeechSession:
        """Retrieve a copy of an existing session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
            return session.model_copy()

    def transition_state(self, session_id: str, new_state: SessionState) -> SpeechSession:
        """Transition a session to a new state if valid."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
                
            current_state = session.state
            if new_state not in self.valid_transitions[current_state]:
                raise InvalidStateTransitionError(
                    f"Cannot transition from {current_state.value} to {new_state.value}."
                )
                
            session.state = new_state
            session.updated_at = _now_utc()
            return session.model_copy()

    def reset_session(self, session_id: str) -> SpeechSession:
        """Reset an active session to the IDLE state."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
                
            if session.state == SessionState.CLOSED:
                raise InvalidStateTransitionError("Cannot reset a CLOSED session.")
                
            session.state = SessionState.IDLE
            session.updated_at = _now_utc()
            return session.model_copy()

    def close_session(self, session_id: str) -> SpeechSession:
        """Close a session, making it terminal."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
                
            if session.state != SessionState.CLOSED:
                session.state = SessionState.CLOSED
                session.updated_at = _now_utc()
            return session.model_copy()
