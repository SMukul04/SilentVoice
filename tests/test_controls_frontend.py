"""Structural tests for the Module 8.7 Frontend Controls (Start/Stop/Restart/Clear)."""

import pytest
from pathlib import Path

@pytest.fixture
def app_js_content():
    """Reads the app.js file."""
    js_path = Path("frontend/static/js/app.js")
    if not js_path.exists():
        pytest.fail("app.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")

def test_start_control_exists(app_js_content):
    """Test 1: Start control exists."""
    assert "btnStartRecognition.addEventListener('click', startCamera)" in app_js_content
    assert "async function startCamera()" in app_js_content

def test_stop_control_exists(app_js_content):
    """Test 2: Stop control exists."""
    assert "btnStopRecognition.addEventListener('click', stopCamera)" in app_js_content
    assert "function stopCamera()" in app_js_content

def test_clear_sentence_exists(app_js_content):
    """Test 3: Clear Sentence exists."""
    assert "btnClearSentence.addEventListener('click'" in app_js_content
    assert "sentenceBuilder.clear()" in app_js_content

def test_start_protected_against_duplicate(app_js_content):
    """Test 4: Start is protected against duplicate initialization."""
    # Ensure startCamera checks if already active
    start_block = app_js_content.split("async function startCamera() {")[1]
    assert "if (isCameraActive) return;" in start_block
    assert "btnStartRecognition.disabled = true;" in start_block

def test_stop_prevents_further_processing(app_js_content):
    """Test 5: Stop prevents further recognition processing."""
    # Ensure stopCamera cancels callbacks
    stop_block = app_js_content.split("function stopCamera() {")[1].split("isCameraActive = false")[0]
    assert "cancelAnimationFrame(animationFrameId)" in stop_block
    assert "cancelVideoFrameCallback" in stop_block
    assert "rVFCId = null" in stop_block

def test_stale_callbacks_cannot_continue(app_js_content):
    """Test 15: Stale callbacks cannot continue recognition after Stop."""
    fetch_then_block = app_js_content.split(".then(data => {")[1].split(".catch(err => {")[0]
    assert "if (!isCameraActive) return;" in fetch_then_block

def test_stop_does_not_clear_sentence(app_js_content):
    """Test 8: Stop does not clear SentenceBuilder."""
    stop_block = app_js_content.split("function stopCamera() {")[1].split("btnStopRecognition.disabled = true;")[0]
    assert "sentenceBuilder.clear()" not in stop_block
    assert "sentenceOutput.textContent = " not in stop_block

def test_webcam_init_failure_safe_state(app_js_content):
    """Test 10: Webcam initialization failure returns UI to safe state."""
    start_catch_block = app_js_content.split("catch (err) {")[1].split("function stopCamera()")[0]
    assert "btnStartRecognition.disabled = false;" in start_catch_block
