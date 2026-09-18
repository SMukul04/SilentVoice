import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.schemas.speech import TextToSpeechResult, SynthesisError
from backend.services.edge_tts_speech_service import EdgeTTSSpeechService
from backend.app.main import app

def test_service_initialization():
    service = EdgeTTSSpeechService(voice="en-GB-SoniaNeural")
    assert service.voice == "en-GB-SoniaNeural"

def test_synthesize_empty_text():
    service = EdgeTTSSpeechService()
    with pytest.raises(ValueError):
        service.synthesize("")
    with pytest.raises(ValueError):
        service.synthesize("   ")

@patch("edge_tts.Communicate")
def test_synthesize_success(mock_communicate):
    mock_comm_instance = MagicMock()
    mock_communicate.return_value = mock_comm_instance
    
    async def mock_stream():
        yield {"type": "audio", "data": b"mock_audio_bytes"}
        
    mock_comm_instance.stream = mock_stream
    
    service = EdgeTTSSpeechService()
    result = service.synthesize("Hello SilentVoice")
    
    assert isinstance(result, TextToSpeechResult)
    assert result.audio_data == b"mock_audio_bytes"
    assert result.content_type == "audio/mpeg"
    assert result.status == "success"
    mock_communicate.assert_called_once_with("Hello SilentVoice", "en-US-AriaNeural")

@patch("edge_tts.Communicate")
def test_synthesize_provider_failure(mock_communicate):
    mock_comm_instance = MagicMock()
    mock_communicate.return_value = mock_comm_instance
    
    async def mock_stream():
        raise Exception("Network error")
        yield {"type": "audio", "data": b""}
        
    mock_comm_instance.stream = mock_stream
    
    service = EdgeTTSSpeechService()
    with pytest.raises(SynthesisError) as excinfo:
        service.synthesize("Hello")
        
    assert "Network error" in str(excinfo.value)

@patch("edge_tts.Communicate")
def test_synthesize_no_audio_failure(mock_communicate):
    mock_comm_instance = MagicMock()
    mock_communicate.return_value = mock_comm_instance
    
    async def mock_stream():
        yield {"type": "WordBoundary", "data": "mock"}
        
    mock_comm_instance.stream = mock_stream
    
    service = EdgeTTSSpeechService()
    with pytest.raises(SynthesisError) as excinfo:
        service.synthesize("Hello")
        
    assert "No audio generated" in str(excinfo.value)


client = TestClient(app)

@patch("backend.api.speech_routes._tts_service.synthesize")
def test_api_synthesize_success(mock_synthesize):
    mock_synthesize.return_value = TextToSpeechResult(
        audio_data=b"test_audio",
        content_type="audio/mpeg",
        status="success"
    )
    
    response = client.post("/api/speech/synthesize", json={"text": "Hello SilentVoice"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"test_audio"

@patch("backend.api.speech_routes._tts_service.synthesize")
def test_api_synthesize_empty_text(mock_synthesize):
    response = client.post("/api/speech/synthesize", json={"text": "   "})
    assert response.status_code == 400

@patch("backend.api.speech_routes._tts_service.synthesize")
def test_api_synthesize_failure(mock_synthesize):
    mock_synthesize.side_effect = SynthesisError("Provider down")
    response = client.post("/api/speech/synthesize", json={"text": "Hello"})
    assert response.status_code == 500
