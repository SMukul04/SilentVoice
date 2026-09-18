"""Edge TTS based speech service implementation.

Provides Text-to-Speech synthesis using the edge-tts package.
Follows the SpeechService interface to decouple the app layer from TTS provider details.
"""

import logging
import asyncio
import threading
from typing import Optional

import edge_tts

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult,
    SpeechError,
    SynthesisError
)
from backend.services.speech_service import SpeechService

logger = logging.getLogger(__name__)


class EdgeTTSSpeechService(SpeechService):
    """Text-to-Speech using edge-tts.
    
    Generates audio in memory using the edge-tts async API.
    """

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        """Initialize the edge-tts service configuration.
        
        Args:
            voice: The voice to use for synthesis. Default is a sensible English voice.
        """
        self.voice = voice
        logger.debug(f"EdgeTTSSpeechService configured with voice={voice}")

    def transcribe(self, audio_input: AudioInput) -> SpeechToTextResult:
        """Speech-to-text is not implemented in this service."""
        raise NotImplementedError("EdgeTTSSpeechService does not support STT.")

    def synthesize(self, text: str) -> TextToSpeechResult:
        """Synthesize text into audio using edge-tts."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace.")

        text = text.strip()

        async def _synthesize_async() -> bytes:
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                audio_data = bytearray()
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data.extend(chunk["data"])
                
                if not audio_data:
                    raise SynthesisError("No audio generated from edge-tts.")
                
                return bytes(audio_data)
            except Exception as e:
                raise SynthesisError(f"Edge TTS provider failed: {str(e)}")

        try:
            # Check if there is already a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # We are running inside an event loop thread. Use a separate thread to run the async code safely.
                result_container = []
                exc_container = []
                
                def _run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result_container.append(new_loop.run_until_complete(_synthesize_async()))
                    except Exception as e:
                        exc_container.append(e)
                    finally:
                        new_loop.close()
                
                t = threading.Thread(target=_run_in_thread)
                t.start()
                t.join()
                
                if exc_container:
                    raise exc_container[0]
                audio_bytes = result_container[0]
            else:
                # No running event loop, we can safely use asyncio.run
                audio_bytes = asyncio.run(_synthesize_async())
                
        except ValueError as e:
            raise e
        except SynthesisError as e:
            logger.error(f"Synthesis failed: {str(e)}")
            raise e
        except Exception as e:
            logger.exception("Unexpected error during synthesis")
            raise SynthesisError(f"Unexpected provider error: {str(e)}")

        return TextToSpeechResult(
            audio_data=audio_bytes,
            content_type="audio/mpeg",
            status="success"
        )
