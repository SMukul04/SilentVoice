"""Abstract speech service interface.

This module establishes the architecture for future speech processing (Module 5.1).
The abstract base class `SpeechService` ensures that the speech subsystem
communicates through clean interfaces so future models/providers (like Whisper,
local TTS, etc.) can be plugged in without redesigning the core system.
"""

from abc import ABC, abstractmethod

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult
)


class SpeechService(ABC):
    """Provider-independent interface for speech operations.
    
    Any concrete implementation (e.g., WhisperSpeechService, CloudSpeechService)
    must inherit from this class and implement these methods. This allows the
    application layer to depend on the interface rather than a specific provider,
    following dependency inversion principles.
    """

    @abstractmethod
    def transcribe(self, audio_input: AudioInput) -> SpeechToTextResult:
        """Transcribe audio into text.
        
        Args:
            audio_input (AudioInput): The abstracted audio input.
            
        Returns:
            SpeechToTextResult: The transcription result.
            
        Raises:
            AudioInputError: If the input is invalid or unsupported.
            TranscriptionError: If the transcription process fails.
        """
        pass

    @abstractmethod
    def synthesize(self, text: str) -> TextToSpeechResult:
        """Synthesize text into audio.
        
        Args:
            text (str): The text to synthesize.
            
        Returns:
            TextToSpeechResult: The synthesized audio result.
            
        Raises:
            ValueError: If the text is empty or invalid.
            SynthesisError: If the synthesis process fails.
        """
        pass
