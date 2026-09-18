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
    SynthesisError
)
from backend.services.whisper_speech_service import WhisperSpeechService
from backend.services.edge_tts_speech_service import EdgeTTSSpeechService

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
