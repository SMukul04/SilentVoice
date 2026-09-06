"""Structural tests for the Prediction Stabilizer Frontend implementation."""

import pytest
from pathlib import Path

@pytest.fixture
def stabilizer_js_content():
    """Reads the prediction_stabilizer.js file."""
    js_path = Path("frontend/static/js/prediction_stabilizer.js")
    if not js_path.exists():
        pytest.fail("prediction_stabilizer.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")

@pytest.fixture
def app_js_content():
    """Reads the app.js file."""
    js_path = Path("frontend/static/js/app.js")
    if not js_path.exists():
        pytest.fail("app.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")

def test_stabilizer_class_exists(stabilizer_js_content):
    """Test 1: PredictionStabilizer class exists."""
    assert "class PredictionStabilizer" in stabilizer_js_content

def test_correct_default_constants(stabilizer_js_content):
    """Test 2: Correct default constants exist."""
    assert "CONFIDENCE_THRESHOLD = 0.60" in stabilizer_js_content
    assert "HISTORY_SIZE = 5" in stabilizer_js_content
    assert "MIN_CONSISTENT_PREDICTIONS = 3" in stabilizer_js_content
    assert "CONFIRMED_PREDICTIONS = 3" in stabilizer_js_content

def test_add_prediction_exists(stabilizer_js_content):
    """Test 3: addPrediction exists."""
    assert "addPrediction(" in stabilizer_js_content

def test_reset_exists(stabilizer_js_content):
    """Test 4: reset exists."""
    assert "reset()" in stabilizer_js_content

def test_history_size_bounded(stabilizer_js_content):
    """Test 5: History size is bounded."""
    assert "this.history.length > this.HISTORY_SIZE" in stabilizer_js_content
    assert "this.history.shift()" in stabilizer_js_content

def test_confidence_threshold_enforced(stabilizer_js_content):
    """Test 6 & 7: Confidence threshold is enforced and boundary behavior represented."""
    assert "confidence >= this.CONFIDENCE_THRESHOLD" in stabilizer_js_content

def test_stability_requires_min_consistent(stabilizer_js_content):
    """Test 8: Stability requires MIN_CONSISTENT_PREDICTIONS and resets when not met."""
    assert "candidateMax >= this.MIN_CONSISTENT_PREDICTIONS" in stabilizer_js_content
    assert "this.stablePrediction = null" in stabilizer_js_content

def test_confirmation_requires_consecutive(stabilizer_js_content):
    """Test 9: Confirmation requires consecutive stable updates."""
    assert "this.stablePredictionCount >=" in stabilizer_js_content
    assert "this.CONFIRMED_PREDICTIONS" in stabilizer_js_content

def test_confirmation_counter_resets_on_change(stabilizer_js_content):
    """Test 10: Confirmation counter resets when stable prediction changes and does not increase when null."""
    assert "this.stablePrediction !== previousStable" in stabilizer_js_content
    assert "this.stablePredictionCount = this.stablePrediction ? 1 : 0" in stabilizer_js_content

def test_temporary_instability_preserves_guard(stabilizer_js_content):
    """Test 10b: Temporary instability does NOT reset lastConfirmedPrediction."""
    # Ensure lastConfirmedPrediction is NOT cleared when stablePrediction changes
    assert "this.lastConfirmedPrediction = null" not in stabilizer_js_content.split("Confirmation logic")[1].split("}")[0]

def test_duplicate_confirmation_prevention(stabilizer_js_content):
    """Test 11: Duplicate confirmation prevention exists."""
    assert "this.confirmedPrediction !== this.lastConfirmedPrediction" in stabilizer_js_content

def test_is_new_confirmation_exposed(stabilizer_js_content):
    """Test 12: isNewConfirmation is exposed."""
    assert "isNewConfirmation:" in stabilizer_js_content

def test_confirmed_maintained_separately(stabilizer_js_content):
    """Test 13: confirmedPrediction is maintained separately from stablePrediction."""
    assert "stablePrediction: this.stablePrediction" in stabilizer_js_content
    assert "confirmedPrediction: this.confirmedPrediction" in stabilizer_js_content

def test_reset_clears_state(stabilizer_js_content):
    """Test 14: reset clears all relevant state."""
    assert "this.history = []" in stabilizer_js_content
    assert "this.stablePrediction = null" in stabilizer_js_content
    assert "this.confirmedPrediction = null" in stabilizer_js_content
    assert "this.stablePredictionCount = 0" in stabilizer_js_content
    assert "this.lastConfirmedPrediction = null" in stabilizer_js_content

def test_app_instantiates_stabilizer(app_js_content):
    """Test 15: app.js instantiates PredictionStabilizer."""
    assert "new PredictionStabilizer()" in app_js_content

def test_app_feeds_predictions(app_js_content):
    """Test 16: app.js feeds backend predictions into it."""
    assert "stabilizer.addPrediction(data.predicted_class, data.confidence)" in app_js_content

def test_app_handles_ui_states(app_js_content):
    """Test 17: app.js handles confirmed/stable UI states."""
    assert '"✓ " + result.confirmedPrediction' in app_js_content
    assert '"Potential sign: " + result.stablePrediction' in app_js_content
    assert '"Analyzing..."' in app_js_content

def test_stop_no_hand_resets(app_js_content):
    """Test 18: stop/no-hand handling resets the stabilizer."""
    assert "stabilizer.reset()" in app_js_content
    assert "extracted.handsDetected === 0" in app_js_content

def test_sentence_building_now_delegated(app_js_content):
    """Test 19: Sentence-building logic is delegated to SentenceBuilder."""
    assert "sentenceBuilder" in app_js_content.split("isPredictionRequestPending")[1]

def test_no_second_predict_loop(app_js_content):
    """Test 20: No second /predict loop is introduced."""
    assert app_js_content.count("fetch('/predict'") == 1 or app_js_content.count("fetch(\"/predict\"") == 1
