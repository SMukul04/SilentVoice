import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_create_session():
    response = client.post("/api/text-to-sign/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["state"] == "IDLE"

def test_get_session():
    # Create first
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    # Get
    get_resp = client.get(f"/api/text-to-sign/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id

def test_get_missing_session():
    response = client.get("/api/text-to-sign/sessions/invalid_id")
    assert response.status_code == 404

def test_process_text():
    # Create session
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    # Process
    process_resp = client.post(
        f"/api/text-to-sign/sessions/{session_id}/process",
        json={"text": "hello please yes"}
    )
    assert process_resp.status_code == 200
    data = process_resp.json()
    assert data["session_id"] == session_id
    assert data["state"] == "COMPLETED"
    assert data["original_text"] == "hello please yes"
    assert len(data["sequence"]) > 0

def test_process_empty_text():
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    process_resp = client.post(
        f"/api/text-to-sign/sessions/{session_id}/process",
        json={"text": "   "}
    )
    assert process_resp.status_code == 400

def test_process_closed_session():
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    client.post(f"/api/text-to-sign/sessions/{session_id}/close")
    
    process_resp = client.post(
        f"/api/text-to-sign/sessions/{session_id}/process",
        json={"text": "hello"}
    )
    assert process_resp.status_code == 409

def test_reset_session():
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    # Simulate processing then error or completed
    client.post(f"/api/text-to-sign/sessions/{session_id}/process", json={"text": "hello"})
    
    reset_resp = client.post(f"/api/text-to-sign/sessions/{session_id}/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["state"] == "IDLE"

def test_close_session():
    create_resp = client.post("/api/text-to-sign/sessions")
    session_id = create_resp.json()["session_id"]
    
    close_resp = client.post(f"/api/text-to-sign/sessions/{session_id}/close")
    assert close_resp.status_code == 200
    assert close_resp.json()["state"] == "CLOSED"
