import pytest
from pydantic import ValidationError

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult,
    SpeechError,
    AudioInputError,
    TranscriptionError,
    SynthesisError
)
from backend.services.speech_service import SpeechService


def test_speech_service_is_abstract():
    """Verify that the SpeechService cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        SpeechService()
    assert "Can't instantiate abstract class SpeechService" in str(exc_info.value)


def test_audio_input_instantiation():
    """Verify that AudioInput behaves correctly with valid and invalid data."""
    # Valid instantiation
    audio = AudioInput(data=b"mock_audio", content_type="audio/wav", sample_rate=16000)
    assert audio.data == b"mock_audio"
    assert audio.content_type == "audio/wav"
    assert audio.sample_rate == 16000

    # Test defaults
    audio_default = AudioInput(data=b"mock_audio")
    assert audio_default.content_type == "audio/wav"
    assert audio_default.sample_rate is None

    # Invalid instantiation (missing required field)
    with pytest.raises(ValidationError):
        AudioInput()


def test_speech_to_text_result_instantiation():
    """Verify that SpeechToTextResult behaves correctly."""
    result = SpeechToTextResult(
        text="Hello world",
        language="en",
        confidence=0.95
    )
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.confidence == 0.95
    assert result.status == "success"

    # Test confidence validation
    with pytest.raises(ValidationError):
        SpeechToTextResult(text="Bad confidence", confidence=1.5)  # > 1.0

    with pytest.raises(ValidationError):
        SpeechToTextResult(text="Bad confidence", confidence=-0.1) # < 0.0


def test_text_to_speech_result_instantiation():
    """Verify that TextToSpeechResult behaves correctly."""
    result = TextToSpeechResult(
        audio_data=b"mock_generated_audio",
        duration_seconds=2.5
    )
    assert result.audio_data == b"mock_generated_audio"
    assert result.content_type == "audio/wav"
    assert result.duration_seconds == 2.5
    assert result.status == "success"


def test_custom_exceptions():
    """Verify that provider-independent exceptions exist and inherit properly."""
    assert issubclass(SpeechError, Exception)
    assert issubclass(AudioInputError, SpeechError)
    assert issubclass(TranscriptionError, SpeechError)
    assert issubclass(SynthesisError, SpeechError)

    # They should be distinct classes
    assert AudioInputError is not TranscriptionError
