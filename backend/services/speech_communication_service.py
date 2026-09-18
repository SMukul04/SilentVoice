"""Speech Communication Orchestration Service.

Integration layer that coordinates the existing speech session, STT, and TTS services
without embedding provider-specific logic.
"""

import logging
from typing import Tuple

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult
)
from backend.schemas.speech_session import SessionState
from backend.schemas.speech_integration import SessionTranscriptionResponse
from backend.services.speech_service import SpeechService
from backend.services.speech_session_service import SpeechSessionService

logger = logging.getLogger(__name__)


class SpeechCommunicationService:
    """Orchestrates speech interactions within a session."""

    def __init__(
        self,
        session_service: SpeechSessionService,
        stt_service: SpeechService,
        tts_service: SpeechService
    ) -> None:
        """Initialize with required services."""
        self.session_service = session_service
        self.stt_service = stt_service
        self.tts_service = tts_service

    def transcribe_session(self, session_id: str, audio_input: AudioInput) -> SessionTranscriptionResponse:
        """Process STT within the context of a session.
        
        Transitions:
        [IDLE] -> LISTENING -> TRANSCRIBING -> COMPLETED
        """
        # Ensure session exists and get current state
        session = self.session_service.get_session(session_id)
        
        if session.state == SessionState.IDLE:
            self.session_service.transition_state(session_id, SessionState.LISTENING)
            
        self.session_service.transition_state(session_id, SessionState.TRANSCRIBING)
        
        try:
            transcription = self.stt_service.transcribe(audio_input)
            updated_session = self.session_service.transition_state(session_id, SessionState.COMPLETED)
            return SessionTranscriptionResponse(
                session=updated_session,
                transcription=transcription
            )
        except Exception:
            self.session_service.transition_state(session_id, SessionState.ERROR)
            raise

    def synthesize_session(self, session_id: str, text: str) -> TextToSpeechResult:
        """Process TTS within the context of a session.
        
        Transitions:
        COMPLETED -> SYNTHESIZING -> COMPLETED
        """
        self.session_service.transition_state(session_id, SessionState.SYNTHESIZING)
        
        try:
            result = self.tts_service.synthesize(text)
            self.session_service.transition_state(session_id, SessionState.COMPLETED)
            return result
        except Exception:
            self.session_service.transition_state(session_id, SessionState.ERROR)
            raise
