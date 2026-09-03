"""Structural tests for the MediaPipe Landmark Extraction Frontend implementation."""

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


@pytest.fixture
def extractor_js_content():
    """Reads the landmark_extractor.js file."""
    js_path = Path("frontend/static/js/landmark_extractor.js")
    if not js_path.exists():
        pytest.fail("landmark_extractor.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")


def test_extractor_module_exists(extractor_js_content):
    """Test 1: Landmark extraction module exists."""
    assert "class LandmarkExtractor" in extractor_js_content


def test_mediapipe_initialization_exists(app_js_content):
    """Test 2: MediaPipe HandLandmarker initialization exists."""
    assert "HandLandmarker.createFromOptions" in app_js_content


def test_model_asset_path(app_js_content):
    """Test 3: Existing hand_landmarker.task asset is referenced."""
    assert "hand_landmarker.task" in app_js_content


def test_21_landmarks_processed(extractor_js_content):
    """Test 4: Exactly 21 landmarks per hand are processed."""
    assert "handLandmarks.length !== 21" in extractor_js_content or "handLandmarks.length != 21" in extractor_js_content
    assert "i < 21" in extractor_js_content


def test_xyz_coordinates(extractor_js_content):
    """Test 5: Each landmark uses x/y/z."""
    assert ".x" in extractor_js_content
    assert ".y" in extractor_js_content
    assert ".z" in extractor_js_content


def test_126_value_output(extractor_js_content):
    """Test 6: Output contains exactly 126 values."""
    assert "FEATURE_DIMENSION = 126" in extractor_js_content
    assert "Float32Array(this.FEATURE_DIMENSION)" in extractor_js_content


def test_missing_hand_zero_filled(extractor_js_content):
    """Test 7: Missing hand is zero-filled."""
    assert "Float32Array(this.HAND_DIMENSION)" in extractor_js_content
    assert "result.features.set(" in extractor_js_content


def test_no_hand_state(extractor_js_content):
    """Test 8: No-hand state is handled."""
    assert "handsDetected = 0" in extractor_js_content
    assert "if (!detectionResult || !detectionResult.landmarks || detectionResult.landmarks.length === 0)" in extractor_js_content


def test_non_finite_values_rejected(extractor_js_content):
    """Test 9: Non-finite values are rejected/normalized."""
    assert "Number.isFinite" in extractor_js_content


def test_left_right_ordering(extractor_js_content):
    """Test 10: Left/Right ordering follows the existing training convention."""
    # Ensure left goes to offset 0, right goes to offset 63
    assert "leftNorm, 0" in extractor_js_content
    assert "rightNorm, this.HAND_DIMENSION" in extractor_js_content


def test_video_timing_loop(app_js_content):
    """Test 11: Processing loop uses video timing/requestAnimationFrame safely."""
    assert "requestAnimationFrame" in app_js_content
    assert "lastVideoTime !==" in app_js_content


def test_processing_stops(app_js_content):
    """Test 12: Processing stops when recognition stops."""
    assert "cancelAnimationFrame" in app_js_content


def test_no_predict_call(extractor_js_content):
    """Test 13: No /predict call is introduced in Module 8.3."""
    assert "fetch('/predict'" not in extractor_js_content
    assert "fetch(\"/predict\"" not in extractor_js_content


def test_hands_detected_exposed(extractor_js_content):
    """Test 15: handsDetected is exposed by the landmark extractor."""
    assert "handsDetected" in extractor_js_content


def test_hands_detected_values(extractor_js_content):
    """Test 16: Zero, one, and two hand results are represented correctly."""
    assert "handsDetected = 0" in extractor_js_content
    # The actual hands count is derived from the detection array length
    assert "detectionResult.landmarks.length" in extractor_js_content


def test_hands_detected_ui_connected(app_js_content, index_html_content):
    """Test 17: The value is connected to the frontend UI and not hardcoded."""
    assert "handsDetectedText" in index_html_content
    assert "handsDetectedText.textContent" in app_js_content
    assert "extracted.handsDetected" in app_js_content
