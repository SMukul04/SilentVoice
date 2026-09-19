"""Text-to-Sign session management service.

Provides a session lifecycle layer for Text-to-Sign operations.
"""

import threading
from typing import Dict
from datetime import datetime, timezone
import uuid

from backend.schemas.text_sign_session import TextSignSession, TextSignSessionState, SessionNotFoundError, InvalidStateTransitionError

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

class TextSignSessionService:
    """Manages Text-to-Sign session lifecycles and state transitions."""

    def __init__(self):
        self._sessions: Dict[str, TextSignSession] = {}
        self._lock = threading.Lock()
        
        # Define allowed state transitions
        self.valid_transitions = {
            TextSignSessionState.IDLE: {
                TextSignSessionState.PROCESSING, 
                TextSignSessionState.CLOSED
            },
            TextSignSessionState.PROCESSING: {
                TextSignSessionState.COMPLETED, 
                TextSignSessionState.ERROR, 
                TextSignSessionState.CLOSED
            },
            TextSignSessionState.COMPLETED: {
                TextSignSessionState.IDLE, 
                TextSignSessionState.CLOSED
            },
            TextSignSessionState.ERROR: {
                TextSignSessionState.IDLE,
                TextSignSessionState.CLOSED
            },
            TextSignSessionState.CLOSED: set()  # Terminal state
        }

    def create_session(self) -> TextSignSession:
        """Create a new session with a unique ID."""
        with self._lock:
            session_id = str(uuid.uuid4())
            session = TextSignSession(
                session_id=session_id,
                state=TextSignSessionState.IDLE,
                created_at=_now_utc(),
                updated_at=_now_utc()
            )
            self._sessions[session_id] = session
            return session.model_copy()

    def get_session(self, session_id: str) -> TextSignSession:
        """Retrieve a copy of an existing session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
            return session.model_copy()

    def transition_state(self, session_id: str, new_state: TextSignSessionState) -> TextSignSession:
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

    def reset_session(self, session_id: str) -> TextSignSession:
        """Reset an active session to the IDLE state."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
                
            if session.state == TextSignSessionState.CLOSED:
                raise InvalidStateTransitionError("Cannot reset a CLOSED session.")
                
            session.state = TextSignSessionState.IDLE
            session.updated_at = _now_utc()
            return session.model_copy()

    def close_session(self, session_id: str) -> TextSignSession:
        """Close a session, making it terminal."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found.")
                
            if session.state != TextSignSessionState.CLOSED:
                session.state = TextSignSessionState.CLOSED
                session.updated_at = _now_utc()
            return session.model_copy()
