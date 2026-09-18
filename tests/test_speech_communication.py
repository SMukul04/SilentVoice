"""Tests for Speech/Text Integration Orchestration."""

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.schemas.speech import (
    AudioInput,
    SpeechToTextResult,
    TextToSpeechResult,
    TranscriptionError,
    SynthesisError,
    AudioInputError
)
from backend.schemas.speech_session import SessionState
from backend.services.speech_session_service import SpeechSessionService
from backend.services.speech_communication_service import SpeechCommunicationService
from backend.app.main import app

def test_communication_service_initialization():
    session_service = SpeechSessionService()
    stt_service = MagicMock()
    tts_service = MagicMock()
    
    comm_service = SpeechCommunicationService(session_service, stt_service, tts_service)
    
    assert comm_service.session_service == session_service
    assert comm_service.stt_service == stt_service
    assert comm_service.tts_service == tts_service

def test_successful_transcription_flow():
    session_service = SpeechSessionService()
    stt_service = MagicMock()
    stt_service.transcribe.return_value = SpeechToTextResult(
        text="Hello", language="en", confidence=1.0, status="success"
    )
    
    comm_service = SpeechCommunicationService(session_service, stt_service, MagicMock())
    session = session_service.create_session()
    
    audio = AudioInput(data=b"dummy", content_type="audio/wav")
    result = comm_service.transcribe_session(session.session_id, audio)
    
    assert result.transcription.text == "Hello"
    assert result.session.state == SessionState.COMPLETED
    assert session_service.get_session(session.session_id).state == SessionState.COMPLETED

def test_successful_synthesis_flow():
    session_service = SpeechSessionService()
    tts_service = MagicMock()
    tts_service.synthesize.return_value = TextToSpeechResult(
        audio_data=b"audio", content_type="audio/mpeg", status="success"
    )
    
    comm_service = SpeechCommunicationService(session_service, MagicMock(), tts_service)
    session = session_service.create_session()
    
    # Must transition to COMPLETED first to synthesize
    session_service.transition_state(session.session_id, SessionState.LISTENING)
    session_service.transition_state(session.session_id, SessionState.TRANSCRIBING)
    session_service.transition_state(session.session_id, SessionState.COMPLETED)
    
    result = comm_service.synthesize_session(session.session_id, "Hello")
    
    assert result.audio_data == b"audio"
    assert session_service.get_session(session.session_id).state == SessionState.COMPLETED

def test_stt_failure():
    session_service = SpeechSessionService()
    stt_service = MagicMock()
    stt_service.transcribe.side_effect = TranscriptionError("Provider failed")
    
    comm_service = SpeechCommunicationService(session_service, stt_service, MagicMock())
    session = session_service.create_session()
    
    audio = AudioInput(data=b"dummy", content_type="audio/wav")
    with pytest.raises(TranscriptionError):
        comm_service.transcribe_session(session.session_id, audio)
        
    assert session_service.get_session(session.session_id).state == SessionState.ERROR

def test_tts_failure():
    session_service = SpeechSessionService()
    tts_service = MagicMock()
    tts_service.synthesize.side_effect = SynthesisError("Provider failed")
    
    comm_service = SpeechCommunicationService(session_service, MagicMock(), tts_service)
    session = session_service.create_session()
    
    session_service.transition_state(session.session_id, SessionState.LISTENING)
    session_service.transition_state(session.session_id, SessionState.TRANSCRIBING)
    session_service.transition_state(session.session_id, SessionState.COMPLETED)
    
    with pytest.raises(SynthesisError):
        comm_service.synthesize_session(session.session_id, "Hello")
        
    assert session_service.get_session(session.session_id).state == SessionState.ERROR


# ==============================================================================
# API TESTS
# ==============================================================================

client = TestClient(app)

@patch("backend.api.speech_routes._speech_service.transcribe")
def test_api_session_transcribe_success(mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="API text", language="en", confidence=0.9
    )
    
    # Create session
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    resp_transcribe = client.post(
        f"/api/speech/sessions/{sid}/transcribe",
        files={"file": ("test.wav", b"dummy", "audio/wav")}
    )
    assert resp_transcribe.status_code == 200
    data = resp_transcribe.json()
    assert data["transcription"]["text"] == "API text"
    assert data["session"]["state"] == "COMPLETED"

@patch("backend.api.speech_routes._tts_service.synthesize")
def test_api_session_synthesize_success(mock_synthesize):
    mock_synthesize.return_value = TextToSpeechResult(
        audio_data=b"audio", content_type="audio/mpeg"
    )
    
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    # Transition to COMPLETED
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "LISTENING"})
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "TRANSCRIBING"})
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "COMPLETED"})
    
    resp_synth = client.post(
        f"/api/speech/sessions/{sid}/synthesize",
        json={"text": "Hello"}
    )
    assert resp_synth.status_code == 200
    assert resp_synth.content == b"audio"
    assert resp_synth.headers["x-session-id"] == sid
    assert resp_synth.headers["x-session-state"] == "COMPLETED"

def test_api_session_transcribe_unknown_session():
    resp = client.post(
        "/api/speech/sessions/invalid/transcribe",
        files={"file": ("test.wav", b"dummy", "audio/wav")}
    )
    assert resp.status_code == 404

def test_api_session_transcribe_closed_session():
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    client.post(f"/api/speech/sessions/{sid}/close")
    
    resp2 = client.post(
        f"/api/speech/sessions/{sid}/transcribe",
        files={"file": ("test.wav", b"dummy", "audio/wav")}
    )
    assert resp2.status_code == 400

@patch("backend.api.speech_routes._speech_service.transcribe")
def test_api_standalone_endpoints_untouched(mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="Standalone", language="en", confidence=0.9
    )
    
    resp = client.post(
        "/api/speech/transcribe",
        files={"file": ("test.wav", b"dummy", "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Standalone"
