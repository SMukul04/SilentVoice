"""Data contracts and models for the speech architecture.

This module defines provider-independent schemas for speech operations,
allowing future STT and TTS implementations (like Whisper, Azure, etc.)
to plug in without changing the communication layer.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Exceptions
# ==============================================================================

class SpeechError(Exception):
    """Base exception for all speech-related errors."""
    pass

class AudioInputError(SpeechError):
    """Raised when the provided audio input is invalid or unsupported."""
    pass

class TranscriptionError(SpeechError):
    """Raised when Speech-to-Text transcription fails."""
    pass

class SynthesisError(SpeechError):
    """Raised when Text-to-Speech synthesis fails."""
    pass


# ==============================================================================
# Models
# ==============================================================================

class AudioInput(BaseModel):
    """An abstraction representing audio input data.
    
    This avoids coupling the backend to a specific audio library. Future
    implementations might extend this or process it from various sources
    (microphone, uploaded file, browser stream).
    """
    data: bytes = Field(..., description="Raw audio bytes.")
    content_type: str = Field("audio/wav", description="MIME type of the audio.")
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz, if known.")


class SpeechToTextResult(BaseModel):
    """Result contract for a speech-to-text operation."""
    text: str = Field(..., description="The transcribed text.")
    language: Optional[str] = Field(None, description="Detected language code (e.g., 'en').")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transcription confidence (0.0 to 1.0).")
    status: str = Field("success", description="Status of the transcription operation.")


class TextToSpeechResult(BaseModel):
    """Result contract for a text-to-speech operation."""
    audio_data: bytes = Field(..., description="The generated audio bytes.")
    content_type: str = Field("audio/wav", description="MIME type of the generated audio.")
    duration_seconds: Optional[float] = Field(None, description="Duration of the generated audio in seconds.")
    status: str = Field("success", description="Status of the synthesis operation.")
