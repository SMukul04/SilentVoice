"""API routes for Text-to-Sign processing."""

import logging
from fastapi import APIRouter, HTTPException

from backend.schemas.text_sign_session import (
    TextSignSession, 
    TextSignSessionState,
    SessionNotFoundError,
    InvalidStateTransitionError
)
from backend.schemas.text_sign import TextSignRequest, TextSignResponse
from backend.schemas.text_processing import EmptyTextError
from backend.services.text_sign_session_service import TextSignSessionService
from backend.services.text_sign_orchestrator_service import TextSignOrchestratorService
from backend.services.text_processing_service import TextProcessingService
from backend.services.sign_resolution_service import SignResolutionService
from backend.services.sign_sequence_service import SignSequenceService
from backend.services.sign_dictionary_service import SignDictionaryService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/text-to-sign",
    tags=["text-to-sign"]
)

# Global service instances
_dictionary_service = SignDictionaryService()
_text_processing_service = TextProcessingService()
_resolution_service = SignResolutionService(dictionary_service=_dictionary_service)
_sequence_service = SignSequenceService()
_session_service = TextSignSessionService()

_orchestrator = TextSignOrchestratorService(
    session_service=_session_service,
    text_processing_service=_text_processing_service,
    resolution_service=_resolution_service,
    sequence_service=_sequence_service
)


@router.post("/sessions", response_model=TextSignSession)
def create_session():
    """Create a new Text-to-Sign session."""
    return _session_service.create_session()


@router.get("/sessions/{session_id}", response_model=TextSignSession)
def get_session(session_id: str):
    """Retrieve an existing Text-to-Sign session."""
    try:
        return _session_service.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sessions/{session_id}/process", response_model=TextSignResponse)
def process_text(session_id: str, request: TextSignRequest):
    """Process text into a sign sequence for the given session."""
    try:
        return _orchestrator.process_text(session_id, request.text)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        # e.g., session is CLOSED or already PROCESSING
        raise HTTPException(status_code=409, detail=str(e))
    except EmptyTextError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during text-to-sign process")
        raise HTTPException(status_code=500, detail="Internal server error during text-to-sign processing.")


@router.post("/sessions/{session_id}/reset", response_model=TextSignSession)
def reset_session(session_id: str):
    """Reset an active session to IDLE."""
    try:
        return _session_service.reset_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/sessions/{session_id}/close", response_model=TextSignSession)
def close_session(session_id: str):
    """Close a session."""
    try:
        return _session_service.close_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
