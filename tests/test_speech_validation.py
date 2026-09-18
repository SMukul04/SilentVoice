"""Integration and Validation suite for Phase 5 (Speech Subsystem)."""

import pytest
from unittest.mock import patch
import concurrent.futures
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.schemas.speech import SpeechToTextResult, TextToSpeechResult, TranscriptionError, SynthesisError
from backend.schemas.speech_session import SessionState

client = TestClient(app)

# ==============================================================================
# 1. VALIDATE STANDALONE STT
# ==============================================================================
@patch("backend.api.speech_routes._speech_service.transcribe")
def test_standalone_stt_valid_audio(mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="Validated text", language="en", confidence=0.95
    )
    resp = client.post(
        "/api/speech/transcribe",
        files={"file": ("audio.wav", b"dummy audio content", "audio/wav")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Validated text"
    assert data["language"] == "en"
    assert data["confidence"] == 0.95

def test_standalone_stt_empty_audio():
    resp = client.post(
        "/api/speech/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()

@patch("backend.api.speech_routes._speech_service.transcribe")
def test_standalone_stt_provider_failure(mock_transcribe):
    mock_transcribe.side_effect = TranscriptionError("Whisper crashed")
    resp = client.post(
        "/api/speech/transcribe",
        files={"file": ("audio.wav", b"dummy audio content", "audio/wav")}
    )
    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()


# ==============================================================================
# 2. VALIDATE STANDALONE TTS
# ==============================================================================
@patch("backend.api.speech_routes._tts_service.synthesize")
def test_standalone_tts_valid_text(mock_synthesize):
    mock_synthesize.return_value = TextToSpeechResult(
        audio_data=b"synthesized audio", content_type="audio/mpeg"
    )
    resp = client.post(
        "/api/speech/synthesize",
        json={"text": "Hello, this is SilentVoice speaking."}
    )
    assert resp.status_code == 200
    assert resp.content == b"synthesized audio"
    assert resp.headers["content-type"] == "audio/mpeg"
    mock_synthesize.assert_called_once_with("Hello, this is SilentVoice speaking.")

def test_standalone_tts_empty_text():
    resp = client.post(
        "/api/speech/synthesize",
        json={"text": "   "}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()

def test_standalone_tts_invalid_json():
    # Sending malformed JSON to trigger RequestValidationError
    resp = client.post(
        "/api/speech/synthesize",
        content="invalid json payload",
        headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "Validation error"
    assert data["detail"] == "Invalid request payload"


@patch("backend.api.speech_routes._tts_service.synthesize")
def test_standalone_tts_provider_failure(mock_synthesize):
    mock_synthesize.side_effect = SynthesisError("Edge TTS failed")
    resp = client.post(
        "/api/speech/synthesize",
        json={"text": "Hello"}
    )
    assert resp.status_code == 500


# ==============================================================================
# 3. VALIDATE SESSION LIFECYCLE
# ==============================================================================
def test_session_lifecycle():
    # CREATE -> IDLE
    resp = client.post("/api/speech/sessions")
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    assert resp.json()["state"] == SessionState.IDLE.value
    
    # -> LISTENING
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.LISTENING.value})
    assert resp.json()["state"] == SessionState.LISTENING.value
    
    # -> TRANSCRIBING
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.TRANSCRIBING.value})
    assert resp.json()["state"] == SessionState.TRANSCRIBING.value
    
    # -> COMPLETED
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.COMPLETED.value})
    assert resp.json()["state"] == SessionState.COMPLETED.value
    
    # -> SYNTHESIZING
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.SYNTHESIZING.value})
    assert resp.json()["state"] == SessionState.SYNTHESIZING.value
    
    # -> COMPLETED
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.COMPLETED.value})
    
    # -> CLOSED
    resp = client.post(f"/api/speech/sessions/{sid}/close")
    assert resp.json()["state"] == SessionState.CLOSED.value
    
    # CLOSED is terminal
    resp = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.IDLE.value})
    assert resp.status_code == 400


# ==============================================================================
# 4. VALIDATE SESSION-AWARE STT
# ==============================================================================
@patch("backend.api.speech_routes._speech_service.transcribe")
def test_session_aware_stt_success(mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="Session aware", language="en", confidence=0.9
    )
    
    # Create session
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    resp_stt = client.post(
        f"/api/speech/sessions/{sid}/transcribe",
        files={"file": ("test.wav", b"audio data", "audio/wav")}
    )
    
    assert resp_stt.status_code == 200
    data = resp_stt.json()
    assert data["transcription"]["text"] == "Session aware"
    assert data["session"]["state"] == SessionState.COMPLETED.value
    assert mock_transcribe.call_count == 1


# ==============================================================================
# 5. VALIDATE SESSION-AWARE TTS
# ==============================================================================
@patch("backend.api.speech_routes._tts_service.synthesize")
def test_session_aware_tts_success(mock_synthesize):
    mock_synthesize.return_value = TextToSpeechResult(
        audio_data=b"session audio", content_type="audio/mpeg"
    )
    
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    # Manually transition to COMPLETED as required by state machine
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.LISTENING.value})
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.TRANSCRIBING.value})
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": SessionState.COMPLETED.value})
    
    resp_tts = client.post(
        f"/api/speech/sessions/{sid}/synthesize",
        json={"text": "Synthesize this"}
    )
    
    assert resp_tts.status_code == 200
    assert resp_tts.content == b"session audio"
    assert resp_tts.headers["x-session-id"] == sid
    assert resp_tts.headers["x-session-state"] == SessionState.COMPLETED.value
    assert mock_synthesize.call_count == 1


# ==============================================================================
# 6. CROSS-PROVIDER FLOW
# ==============================================================================
@patch("backend.api.speech_routes._speech_service.transcribe")
@patch("backend.api.speech_routes._tts_service.synthesize")
def test_cross_provider_flow(mock_synthesize, mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="Flow test", language="en", confidence=0.9
    )
    mock_synthesize.return_value = TextToSpeechResult(
        audio_data=b"flow audio", content_type="audio/mpeg"
    )
    
    # 1. Create Session
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    # 2. STT
    resp_stt = client.post(
        f"/api/speech/sessions/{sid}/transcribe",
        files={"file": ("test.wav", b"audio data", "audio/wav")}
    )
    assert resp_stt.status_code == 200
    text_result = resp_stt.json()["transcription"]["text"]
    
    # 3. TTS using resulting text
    resp_tts = client.post(
        f"/api/speech/sessions/{sid}/synthesize",
        json={"text": text_result}
    )
    assert resp_tts.status_code == 200
    assert resp_tts.content == b"flow audio"
    
    # Verify mock arguments
    mock_synthesize.assert_called_once_with("Flow test")
    
    # Verify final state
    resp_get = client.get(f"/api/speech/sessions/{sid}")
    assert resp_get.json()["state"] == SessionState.COMPLETED.value


# ==============================================================================
# 7. ERROR RECOVERY
# ==============================================================================
@patch("backend.api.speech_routes._speech_service.transcribe")
def test_error_state_recovery(mock_transcribe):
    mock_transcribe.side_effect = TranscriptionError("Failed")
    
    resp = client.post("/api/speech/sessions")
    sid = resp.json()["session_id"]
    
    resp_stt = client.post(
        f"/api/speech/sessions/{sid}/transcribe",
        files={"file": ("test.wav", b"audio data", "audio/wav")}
    )
    assert resp_stt.status_code == 500
    
    # State should be ERROR
    resp_get = client.get(f"/api/speech/sessions/{sid}")
    assert resp_get.json()["state"] == SessionState.ERROR.value
    
    # Reset to IDLE
    resp_reset = client.post(f"/api/speech/sessions/{sid}/reset")
    assert resp_reset.status_code == 200
    assert resp_reset.json()["state"] == SessionState.IDLE.value


# ==============================================================================
# 8. CONCURRENCY & ISOLATION
# ==============================================================================
@patch("backend.api.speech_routes._speech_service.transcribe")
def test_session_concurrency_isolation(mock_transcribe):
    mock_transcribe.return_value = SpeechToTextResult(
        text="Concurrency", language="en", confidence=0.9
    )
    
    sids = []
    for _ in range(5):
        sids.append(client.post("/api/speech/sessions").json()["session_id"])
        
    def worker(sid):
        return client.post(
            f"/api/speech/sessions/{sid}/transcribe",
            files={"file": ("test.wav", b"audio", "audio/wav")}
        )
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, sid) for sid in sids]
        concurrent.futures.wait(futures)
        
    for sid in sids:
        resp = client.get(f"/api/speech/sessions/{sid}")
        assert resp.json()["state"] == SessionState.COMPLETED.value
        
    # All are unique
    assert len(set(sids)) == 5
