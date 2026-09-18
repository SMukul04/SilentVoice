"""Audio normalization utility for the speech architecture.

Provides an internal boundary layer that sanitizes arbitrary browser audio
(e.g., WebM, OGG, uncompressed WAV) into a format fully supported by STT engines.
"""

import io
import shutil
import logging
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from backend.schemas.speech import AudioInput, AudioInputError

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg or avconv is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None or shutil.which("avconv") is not None


def normalize_audio(audio_input: AudioInput) -> bytes:
    """Normalize the incoming audio to a 16kHz, mono WAV format.
    
    Args:
        audio_input: The AudioInput schema containing raw bytes and content type.
        
    Returns:
        bytes: Raw WAV audio bytes normalized for STT consumption.
        
    Raises:
        AudioInputError: If FFmpeg is missing, or if the audio bytes are invalid/corrupt.
    """
    if not audio_input.data:
        raise AudioInputError("Audio input data is empty.")

    if not is_ffmpeg_available():
        logger.error("FFmpeg is not installed or not found in PATH.")
        raise AudioInputError(
            "System dependency 'ffmpeg' is missing. Please install ffmpeg on the host OS "
            "to process arbitrary audio formats."
        )

    try:
        # Load audio from memory
        audio_io = io.BytesIO(audio_input.data)
        
        # pydub can automatically detect most formats, but passing format helps if we know it
        format_hint = None
        if "webm" in audio_input.content_type.lower():
            format_hint = "webm"
        elif "ogg" in audio_input.content_type.lower():
            format_hint = "ogg"
        elif "wav" in audio_input.content_type.lower():
            format_hint = "wav"
            
        try:
            audio = AudioSegment.from_file(audio_io, format=format_hint)
        except CouldntDecodeError:
            # Fallback without hint if the content_type was misleading
            audio_io.seek(0)
            audio = AudioSegment.from_file(audio_io)

        # Normalize to 16kHz, mono (1 channel)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Export back to bytes as WAV
        out_io = io.BytesIO()
        audio.export(out_io, format="wav")
        
        return out_io.getvalue()
        
    except CouldntDecodeError as e:
        logger.warning(f"Failed to decode audio input: {e}")
        raise AudioInputError("Unsupported or corrupt audio format provided.")
    except Exception as e:
        logger.exception("Unexpected error during audio normalization")
        raise AudioInputError(f"Audio processing failed: {str(e)}")
