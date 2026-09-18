"""API routes for speech processing."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend.schemas.speech import (
    AudioInput, 
    SpeechToTextResult,
    SpeechError,
    AudioInputError,
    TranscriptionError,
    SynthesisError,
    SessionNotFoundError,
    InvalidStateTransitionError
)
from backend.schemas.speech_integration import SessionTranscriptionResponse
from backend.services.speech_communication_service import SpeechCommunicationService
from backend.services.whisper_speech_service import WhisperSpeechService
from backend.services.edge_tts_speech_service import EdgeTTSSpeechService
from backend.schemas.speech_session import SpeechSession, SessionState
from backend.services.speech_session_service import SpeechSessionService, SessionNotFoundError, InvalidStateTransitionError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/speech",
    tags=["speech"]
)

# Global service instance with sensible local defaults
# This persists the model between requests once lazily loaded.
_speech_service = WhisperSpeechService(
    model_size="base",
    device="cpu",
    compute_type="int8"
)


@router.post("/transcribe", response_model=SpeechToTextResult)
async def transcribe_audio(file: UploadFile = File(...)) -> SpeechToTextResult:
    """Transcribe an uploaded audio file into text.
    
    Accepts multipart/form-data containing an audio file. Normalizes
    the audio to ensure compatibility with the underlying STT engine,
    and returns the transcribed text.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
        
    try:
        # Read uploaded bytes
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
            
        # Construct provider-independent input contract
        audio_input = AudioInput(
            data=audio_bytes,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Process transcription
        result = _speech_service.transcribe(audio_input)
        return result
        
    except AudioInputError as e:
        logger.warning(f"Audio input error: {e}")
        # Map provider-independent audio validation to HTTP 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))
        
    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        # Map transcription failures to HTTP 500
        raise HTTPException(status_code=500, detail="Transcription failed. See server logs.")
        
    except SpeechError as e:
        logger.error(f"Speech service error: {e}")
        raise HTTPException(status_code=500, detail="Speech service unavailable.")
        
    except HTTPException:
        # Re-raise standard HTTP exceptions so they return correct status codes
        raise
    except Exception as e:
        logger.exception("Unexpected error during speech route execution")
        raise HTTPException(status_code=500, detail="Internal server error during transcription.")

class SynthesizeRequest(BaseModel):
    text: str

_tts_service = EdgeTTSSpeechService(voice="en-US-AriaNeural")

@router.post("/synthesize")
def synthesize_text(request: SynthesizeRequest) -> Response:
    """Synthesize text into audio."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        result = _tts_service.synthesize(request.text)
        return Response(content=result.audio_data, media_type=result.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SynthesisError as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail="Synthesis failed. See server logs.")
    except Exception as e:
        logger.exception("Unexpected error during synthesis route execution")
        raise HTTPException(status_code=500, detail="Internal server error during synthesis.")

# ==============================================================================
# Speech Communication / Orchestration API
# ==============================================================================

_session_service = SpeechSessionService()

_communication_service = SpeechCommunicationService(
    session_service=_session_service,
    stt_service=_speech_service,
    tts_service=_tts_service
)

@router.post("/sessions/{session_id}/transcribe", response_model=SessionTranscriptionResponse)
async def session_transcribe(session_id: str, file: UploadFile = File(...)):
    """Transcribe audio within a session."""
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
        
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
            
        audio_input = AudioInput(
            data=audio_bytes,
            content_type=file.content_type or "application/octet-stream"
        )
        
        return _communication_service.transcribe_session(session_id, audio_input)
        
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AudioInputError as e:
        logger.warning(f"Audio input error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed. See server logs.")
    except Exception as e:
        logger.exception("Unexpected error during session transcribe")
        raise HTTPException(status_code=500, detail="Internal server error during transcription.")

@router.post("/sessions/{session_id}/synthesize")
def session_synthesize(session_id: str, request: SynthesizeRequest) -> Response:
    """Synthesize text into audio within a session."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        result = _communication_service.synthesize_session(session_id, request.text)
        
        response = Response(content=result.audio_data, media_type=result.content_type)
        response.headers["X-Session-ID"] = session_id
        response.headers["X-Session-State"] = SessionState.COMPLETED.value
        return response
        
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SynthesisError as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail="Synthesis failed. See server logs.")
    except Exception as e:
        logger.exception("Unexpected error during session synthesize")
        raise HTTPException(status_code=500, detail="Internal server error during synthesis.")


# ==============================================================================
# Session Management API
# ==============================================================================

@router.post("/sessions", response_model=SpeechSession)
def create_session():
    """Create a new speech session."""
    return _session_service.create_session()

@router.get("/sessions/{session_id}", response_model=SpeechSession)
def get_session(session_id: str):
    """Retrieve an existing speech session."""
    try:
        return _session_service.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

class TransitionRequest(BaseModel):
    state: SessionState

@router.post("/sessions/{session_id}/transition", response_model=SpeechSession)
def transition_session(session_id: str, request: TransitionRequest):
    """Transition a session to a new state."""
    try:
        return _session_service.transition_state(session_id, request.state)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sessions/{session_id}/reset", response_model=SpeechSession)
def reset_session(session_id: str):
    """Reset an active session to IDLE."""
    try:
        return _session_service.reset_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sessions/{session_id}/close", response_model=SpeechSession)
def close_session(session_id: str):
    """Close a session."""
    try:
        return _session_service.close_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
