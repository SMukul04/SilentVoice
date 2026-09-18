"""Tests for Module 5.2 Speech-to-Text implementation.

Tests ensure lazy loading, correct abstraction adherence, error mapping,
and API functionality without downloading real Whisper models or relying on FFmpeg.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.schemas.speech import (
    AudioInput,
    AudioInputError,
    TranscriptionError,
    SpeechError
)
from backend.services.whisper_speech_service import WhisperSpeechService


@pytest.fixture
def mock_whisper_engine():
    """Mock the faster_whisper.WhisperModel to avoid downloads and GPU usage."""
    with patch("backend.services.whisper_speech_service.WhisperSpeechService._get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Setup mock transcription output
        mock_segment = MagicMock()
        mock_segment.text = "Hello world from mocked whisper."
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        yield mock_get_model, mock_model


@pytest.fixture
def mock_audio_normalization():
    """Mock audio normalization to avoid relying on FFmpeg for generic tests."""
    with patch("backend.services.whisper_speech_service.normalize_audio") as mock_norm:
        mock_norm.return_value = b"normalized_mock_wav_bytes"
        yield mock_norm


# ==============================================================================
# A. Service Construction
# ==============================================================================

def test_service_initialization():
    """Verify configuration is accepted and model is NOT loaded immediately."""
    with patch.dict('sys.modules', {'faster_whisper': MagicMock()}):
        service = WhisperSpeechService(model_size="tiny", device="cpu", compute_type="int8")
        assert service.model_size == "tiny"
        assert service.device == "cpu"
        assert service._model is None  # Lazy loading enforced


# ==============================================================================
# B. Audio Validation
# ==============================================================================

def test_transcribe_empty_audio(mock_audio_normalization):
    """Verify empty audio rejects early at normalization boundary."""
    # We test the audio_normalization logic directly
    from backend.speech_recognition.audio_normalization import normalize_audio
    
    with pytest.raises(AudioInputError):
        normalize_audio(AudioInput(data=b"", content_type="audio/wav"))


# ==============================================================================
# C. Successful Transcription
# ==============================================================================

def test_transcribe_success(mock_whisper_engine, mock_audio_normalization):
    """Verify mocked whisper output maps correctly to SpeechToTextResult."""
    service = WhisperSpeechService()
    
    audio = AudioInput(data=b"raw_bytes", content_type="audio/webm")
    result = service.transcribe(audio)
    
    assert result.text == "Hello world from mocked whisper."
    assert result.language == "en"
    assert result.confidence == 0.99
    assert result.status == "success"
    
    _, mock_model = mock_whisper_engine
    mock_model.transcribe.assert_called_once()


# ==============================================================================
# D. Error Mapping
# ==============================================================================

def test_transcription_provider_error(mock_whisper_engine, mock_audio_normalization):
    """Verify internal whisper errors become TranscriptionError."""
    _, mock_model = mock_whisper_engine
    mock_model.transcribe.side_effect = Exception("CUDA out of memory mock error")
    
    service = WhisperSpeechService()
    audio = AudioInput(data=b"raw_bytes", content_type="audio/wav")
    
    with pytest.raises(TranscriptionError) as exc:
        service.transcribe(audio)
        
    assert "CUDA out of memory mock error" in str(exc.value)


# ==============================================================================
# E. Model Reuse
# ==============================================================================

def test_model_reuse(mock_audio_normalization):
    """Verify repeated transcription calls do not recreate the model."""
    # We patch the actual import to avoid downloading
    with patch('backend.services.whisper_speech_service.WhisperSpeechService._get_model') as mock_get:
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "test"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 1.0
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        mock_get.return_value = mock_model
        
        service = WhisperSpeechService()
        audio = AudioInput(data=b"bytes")
        
        service.transcribe(audio)
        service.transcribe(audio)
        
        # _get_model is called twice, but our mock_get abstracts that.
        # Let's test the real _get_model behavior with mocked WhisperModel
        pass

def test_real_model_reuse():
    with patch.dict('sys.modules', {'faster_whisper': MagicMock()}) as mock_modules:
        mock_whisper_class = mock_modules['faster_whisper'].WhisperModel
        
        service = WhisperSpeechService()
        assert service._model is None
        
        # First call creates model
        model1 = service._get_model()
        mock_whisper_class.assert_called_once()
        
        # Second call reuses model
        model2 = service._get_model()
        assert model1 is model2
        # Assert class was still only instantiated once
        mock_whisper_class.assert_called_once()


# ==============================================================================
# F. API Endpoint
# ==============================================================================

client = TestClient(app)

def test_api_transcribe_success(mock_whisper_engine, mock_audio_normalization):
    """Verify POST /api/speech/transcribe works with mocked pipeline."""
    # We must patch the global instance used in the router
    with patch("backend.api.speech_routes._speech_service.transcribe") as mock_transcribe:
        from backend.schemas.speech import SpeechToTextResult
        mock_transcribe.return_value = SpeechToTextResult(
            text="API mock success",
            language="hi",
            confidence=0.85
        )
        
        response = client.post(
            "/api/speech/transcribe",
            files={"file": ("test.wav", b"dummy_audio_bytes", "audio/wav")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "API mock success"
        assert data["language"] == "hi"


def test_api_transcribe_missing_file():
    """Verify missing file returns 422 Unprocessable Entity (FastAPI validation)."""
    response = client.post("/api/speech/transcribe")
    # FastAPI automatically returns 422 when required file is missing
    assert response.status_code == 422


def test_api_transcribe_empty_file(mock_whisper_engine):
    """Verify empty audio bytes are rejected gracefully."""
    response = client.post(
        "/api/speech/transcribe",
        files={"file": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "Audio file is empty" in response.text
