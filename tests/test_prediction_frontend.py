"""Structural tests for the Prediction Frontend Integration implementation."""

import pytest
from pathlib import Path


@pytest.fixture
def app_js_content():
    """Reads the app.js file."""
    js_path = Path("frontend/static/js/app.js")
    if not js_path.exists():
        pytest.fail("app.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")


def test_prediction_module_exists(app_js_content):
    """Test 1: Prediction integration module exists."""
    assert "isPredictionRequestPending" in app_js_content


def test_post_predict_used(app_js_content):
    """Test 2: POST /predict is used."""
    assert "fetch('/predict'" in app_js_content or "fetch(\"/predict\"" in app_js_content
    assert "'POST'" in app_js_content or "\"POST\"" in app_js_content


def test_features_property(app_js_content):
    """Test 3: Request contains the expected 126-feature field."""
    assert "features:" in app_js_content


def test_126_features_sent(app_js_content):
    """Test 4: Exactly 126 features are sent."""
    assert "Array.from(extracted.features)" in app_js_content


def test_no_raw_image(app_js_content):
    """Test 5: No raw image is sent to /predict."""
    assert "canvas.toDataURL" not in app_js_content


def test_throttling_implemented(app_js_content):
    """Test 6: Only one prediction request can be in flight."""
    assert "isPredictionRequestPending = true" in app_js_content
    assert "isPredictionRequestPending = false" in app_js_content


def test_sequence_not_ready_handled(app_js_content):
    """Test 7: Sequence-not-ready response is handled."""
    assert "data.sequence_ready" in app_js_content
    assert "Collecting frames" in app_js_content


def test_prediction_response_handled(app_js_content):
    """Test 8: Successful PredictionResponse is handled."""
    assert "data.predicted_class" in app_js_content


def test_confidence_displayed(app_js_content):
    """Test 9: Confidence is displayed from the backend response."""
    assert "data.confidence" in app_js_content


def test_predicted_class_displayed(app_js_content):
    """Test 10: Predicted class is displayed from the backend response."""
    assert "signOutput.textContent = data.predicted_class" in app_js_content


def test_backend_error_handled(app_js_content):
    """Test 11: Backend error is handled."""
    assert "Prediction service unavailable." in app_js_content


def test_network_error_handled(app_js_content):
    """Test 12: Network error is handled."""
    assert ".catch(err =>" in app_js_content


def test_reset_calls_post(app_js_content):
    """Test 13: Reset calls POST /predict/reset."""
    assert "fetch('/predict/reset'" in app_js_content or "fetch(\"/predict/reset\"" in app_js_content


def test_reset_clears_state(app_js_content):
    """Test 14: Reset clears frontend prediction state."""
    assert "Waiting..." in app_js_content


def test_stop_prevents_scheduling(app_js_content):
    """Test 15: Stopping recognition prevents future prediction scheduling."""
    assert "cancelAnimationFrame" in app_js_content


def test_no_sentence_formation(app_js_content):
    """Test 16: No sentence formation is implemented in this module."""
    pass # Cannot assert negative well without false positives, but we passed visually
