"""Whisper-based speech service implementation.

Provides local STT transcription using faster-whisper.
Follows the SpeechService interface to decouple the app layer from Whisper details.
"""

import logging
from typing import Optional

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult,
    SpeechError,
    AudioInputError,
    TranscriptionError
)
from backend.services.speech_service import SpeechService
from backend.speech_recognition.audio_normalization import normalize_audio

logger = logging.getLogger(__name__)


class WhisperSpeechService(SpeechService):
    """Local Speech-to-Text using faster-whisper.
    
    The model is loaded lazily on the first transcription request to avoid
    heavy initialization during application startup.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None
    ) -> None:
        """Initialize the whisper service configuration.
        
        Args:
            model_size: Whisper model size (e.g. "tiny", "base", "small").
            device: Compute device ("cpu" or "cuda").
            compute_type: Quantization type (e.g. "int8", "float16").
            language: Default language code (e.g., "en", "hi"). If None, auto-detects.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        logger.debug(f"WhisperSpeechService configured with model={model_size}, device={device}")

    def _get_model(self):
        """Lazy load the WhisperModel."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device}...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.exception("Failed to load faster-whisper model")
                raise SpeechError(f"Model initialization failed: {str(e)}")
        return self._model

    def transcribe(self, audio_input: AudioInput) -> SpeechToTextResult:
        """Transcribe normalized audio using faster-whisper."""
        # 1. Normalize Audio
        try:
            normalized_wav_bytes = normalize_audio(audio_input)
        except AudioInputError as e:
            # Re-raise AudioInputError directly
            raise e
        except Exception as e:
            raise AudioInputError(f"Unexpected error normalizing audio: {str(e)}")

        # 2. Lazy load model
        model = self._get_model()

        # 3. Transcribe
        try:
            # faster_whisper accepts bytes or file path. 
            # We can pass the raw wav bytes by wrapping in io.BytesIO
            import io
            audio_io = io.BytesIO(normalized_wav_bytes)
            
            segments, info = model.transcribe(
                audio_io,
                language=self.language,
                vad_filter=True  # useful for ignoring silence
            )
            
            # segments is a generator, force evaluation to get text
            text_blocks = []
            for segment in segments:
                text_blocks.append(segment.text)
                
            transcribed_text = "".join(text_blocks).strip()
            
            if not transcribed_text:
                raise TranscriptionError("No speech detected or transcription was empty.")

            return SpeechToTextResult(
                text=transcribed_text,
                language=info.language,
                confidence=info.language_probability,
                status="success"
            )
            
        except SpeechError as e:
            raise e
        except Exception as e:
            logger.exception("Transcription failed")
            raise TranscriptionError(f"Whisper transcription failed: {str(e)}")

    def synthesize(self, text: str) -> TextToSpeechResult:
        """Text-to-speech is not implemented in this service."""
        raise NotImplementedError("WhisperSpeechService does not support TTS.")
