"""Tests for Speech Session Management module."""

import pytest
from datetime import datetime
import concurrent.futures

from fastapi.testclient import TestClient

from backend.schemas.speech_session import SessionState, SpeechSession
from backend.schemas.speech import SessionNotFoundError, InvalidStateTransitionError
from backend.services.speech_session_service import SpeechSessionService
from backend.app.main import app

def test_session_creation():
    service = SpeechSessionService()
    session = service.create_session()
    
    assert session.session_id is not None
    assert session.state == SessionState.IDLE
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)

def test_unique_session_ids():
    service = SpeechSessionService()
    s1 = service.create_session()
    s2 = service.create_session()
    
    assert s1.session_id != s2.session_id

def test_get_existing_session():
    service = SpeechSessionService()
    created = service.create_session()
    retrieved = service.get_session(created.session_id)
    
    assert retrieved.session_id == created.session_id
    assert retrieved.state == created.state

def test_unknown_session_handling():
    service = SpeechSessionService()
    with pytest.raises(SessionNotFoundError):
        service.get_session("non-existent-id")
        
def test_valid_state_transitions():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    # IDLE -> LISTENING
    session = service.transition_state(sid, SessionState.LISTENING)
    assert session.state == SessionState.LISTENING
    
    # LISTENING -> TRANSCRIBING
    session = service.transition_state(sid, SessionState.TRANSCRIBING)
    assert session.state == SessionState.TRANSCRIBING
    
    # TRANSCRIBING -> COMPLETED
    session = service.transition_state(sid, SessionState.COMPLETED)
    assert session.state == SessionState.COMPLETED
    
    # COMPLETED -> SYNTHESIZING
    session = service.transition_state(sid, SessionState.SYNTHESIZING)
    assert session.state == SessionState.SYNTHESIZING
    
    # SYNTHESIZING -> COMPLETED
    session = service.transition_state(sid, SessionState.COMPLETED)
    assert session.state == SessionState.COMPLETED

def test_invalid_state_transitions():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    # IDLE -> COMPLETED is invalid
    with pytest.raises(InvalidStateTransitionError):
        service.transition_state(sid, SessionState.COMPLETED)

def test_closed_is_terminal():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    service.transition_state(sid, SessionState.CLOSED)
    
    with pytest.raises(InvalidStateTransitionError):
        service.transition_state(sid, SessionState.IDLE)
        
    with pytest.raises(InvalidStateTransitionError):
        service.transition_state(sid, SessionState.LISTENING)

def test_reset_behavior():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    service.transition_state(sid, SessionState.LISTENING)
    
    res = service.reset_session(sid)
    assert res.state == SessionState.IDLE

def test_close_behavior():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    res = service.close_session(sid)
    assert res.state == SessionState.CLOSED

def test_session_isolation():
    service = SpeechSessionService()
    s1 = service.create_session()
    s2 = service.create_session()
    
    service.transition_state(s1.session_id, SessionState.LISTENING)
    
    ret_s1 = service.get_session(s1.session_id)
    ret_s2 = service.get_session(s2.session_id)
    
    assert ret_s1.state == SessionState.LISTENING
    assert ret_s2.state == SessionState.IDLE

def test_concurrent_mutations():
    service = SpeechSessionService()
    session = service.create_session()
    sid = session.session_id
    
    def worker():
        try:
            service.transition_state(sid, SessionState.LISTENING)
        except InvalidStateTransitionError:
            pass
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        concurrent.futures.wait(futures)
        
    ret = service.get_session(sid)
    assert ret.state == SessionState.LISTENING


# API tests
client = TestClient(app)

def test_api_session_creation():
    response = client.post("/api/speech/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["state"] == "IDLE"

def test_api_session_retrieval():
    resp1 = client.post("/api/speech/sessions")
    sid = resp1.json()["session_id"]
    
    resp2 = client.get(f"/api/speech/sessions/{sid}")
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == sid

def test_api_transition():
    resp1 = client.post("/api/speech/sessions")
    sid = resp1.json()["session_id"]
    
    resp2 = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "LISTENING"})
    assert resp2.status_code == 200
    assert resp2.json()["state"] == "LISTENING"

def test_api_reset():
    resp1 = client.post("/api/speech/sessions")
    sid = resp1.json()["session_id"]
    
    client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "LISTENING"})
    
    resp3 = client.post(f"/api/speech/sessions/{sid}/reset")
    assert resp3.status_code == 200
    assert resp3.json()["state"] == "IDLE"

def test_api_close():
    resp1 = client.post("/api/speech/sessions")
    sid = resp1.json()["session_id"]
    
    resp3 = client.post(f"/api/speech/sessions/{sid}/close")
    assert resp3.status_code == 200
    assert resp3.json()["state"] == "CLOSED"

def test_api_invalid_session():
    resp = client.get("/api/speech/sessions/non-existent")
    assert resp.status_code == 404

def test_api_invalid_transition():
    resp1 = client.post("/api/speech/sessions")
    sid = resp1.json()["session_id"]
    
    resp2 = client.post(f"/api/speech/sessions/{sid}/transition", json={"state": "COMPLETED"})
    assert resp2.status_code == 400
