"""Structural tests for the Webcam Frontend Interface."""

import pytest
from pathlib import Path
import re


@pytest.fixture
def index_html_content():
    """Reads the index.html file."""
    html_path = Path("frontend/templates/index.html")
    if not html_path.exists():
        pytest.fail("index.html not found in frontend/templates/")
    return html_path.read_text(encoding="utf-8")


@pytest.fixture
def app_js_content():
    """Reads the app.js file."""
    js_path = Path("frontend/static/js/app.js")
    if not js_path.exists():
        pytest.fail("app.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")


def test_video_element_exists(index_html_content):
    """Test 2: Video element exists."""
    assert "<video" in index_html_content
    assert "autoplay" in index_html_content
    assert "playsinline" in index_html_content
    assert "muted" in index_html_content
    assert "id=\"webcamVideo\"" in index_html_content


def test_start_button_exists(index_html_content):
    """Test 3: Start button exists."""
    assert "id=\"btnStartRecognition\"" in index_html_content


def test_stop_button_exists(index_html_content):
    """Test 4: Stop button exists."""
    assert "id=\"btnStopRecognition\"" in index_html_content


def test_camera_status_element_exists(index_html_content):
    """Test 5: Camera status element exists."""
    assert "id=\"cameraStatusText\"" in index_html_content


def test_start_camera_logic_exists(app_js_content):
    """Test 6: Start camera logic exists."""
    assert "navigator.mediaDevices.getUserMedia" in app_js_content
    assert "btnStartRecognition.addEventListener('click', startCamera)" in app_js_content


def test_stop_camera_logic_exists(app_js_content):
    """Test 7: Stop camera logic exists."""
    assert "btnStopRecognition.addEventListener('click', stopCamera)" in app_js_content
    assert "track.stop()" in app_js_content


def test_permission_denied_handling_exists(app_js_content):
    """Test 8: Permission-denied handling exists."""
    assert "NotAllowedError" in app_js_content
    assert "Camera permission was denied." in app_js_content


def test_no_camera_handling_exists(app_js_content):
    """Test 9: No-camera handling exists."""
    assert "NotFoundError" in app_js_content
    assert "No camera was found." in app_js_content


def test_cleanup_logic_exists(app_js_content):
    """Test 10: Camera stream tracks are stopped during cleanup."""
    assert "beforeunload" in app_js_content
    assert "track.stop()" in app_js_content


def test_audio_not_requested(app_js_content):
    """Test 11: Audio is not requested."""
    # Ensure audio: false is present in constraints
    assert re.search(r"audio\s*:\s*false", app_js_content), "audio: false not found in constraints"
