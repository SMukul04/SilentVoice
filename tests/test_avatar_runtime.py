"""Tests for the Phase 7.3 Avatar Runtime."""

import pytest

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_runtime_service import (
    MockAvatarRuntimeService,
    AvatarRuntimeState,
    RuntimeNotInitializedError,
    InvalidRuntimeStateError
)


def create_mock_asset(sign_id: str) -> AnimationAsset:
    """Helper to create valid assets for tests."""
    return AnimationAsset(
        sign_id=sign_id,
        asset_id=f"{sign_id.lower()}_default",
        availability=AnimationAvailability.AVAILABLE
    )


def test_runtime_initial_state():
    """Test the runtime starts uninitialized."""
    runtime = MockAvatarRuntimeService()
    assert runtime.get_state() == AvatarRuntimeState.UNINITIALIZED
    assert runtime.get_current_animation() is None


def test_runtime_initialize():
    """Test initialization lifecycle."""
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    assert runtime.get_state() == AvatarRuntimeState.READY
    
    # Re-initialization should be idempotent for READY state
    runtime.initialize()
    assert runtime.get_state() == AvatarRuntimeState.READY


def test_runtime_play_animation():
    """Test playing a single animation."""
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    
    asset = create_mock_asset("HELLO")
    runtime.play_animation(asset)
    
    # Mock completes immediately and returns to READY
    assert runtime.get_state() == AvatarRuntimeState.READY
    assert runtime.get_current_animation() == asset
    
    history = runtime.get_history()
    assert len(history) == 1
    assert history[0].sign_id == "HELLO"


def test_runtime_play_sequence():
    """Test sequences strictly preserve order and repeats."""
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    
    assets = [
        create_mock_asset("HELLO"),
        create_mock_asset("WORLD"),
        create_mock_asset("HELLO"),
    ]
    
    runtime.play_sequence(assets)
    
    assert runtime.get_state() == AvatarRuntimeState.READY
    
    history = runtime.get_history()
    assert len(history) == 3
    assert history[0].sign_id == "HELLO"
    assert history[1].sign_id == "WORLD"
    assert history[2].sign_id == "HELLO"


def test_runtime_stop_lifecycle():
    """Test stopping the runtime."""
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    
    runtime.stop()
    assert runtime.get_state() == AvatarRuntimeState.STOPPED
    assert runtime.get_current_animation() is None


def test_runtime_reset_lifecycle():
    """Test full reset clears history and reverts to uninitialized."""
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    
    runtime.play_animation(create_mock_asset("HELLO"))
    assert len(runtime.get_history()) == 1
    
    runtime.reset()
    assert runtime.get_state() == AvatarRuntimeState.UNINITIALIZED
    assert runtime.get_current_animation() is None
    assert len(runtime.get_history()) == 0


def test_runtime_playback_without_initialization():
    """Test playing before initialization raises explicit error."""
    runtime = MockAvatarRuntimeService()
    
    asset = create_mock_asset("HELLO")
    
    with pytest.raises(RuntimeNotInitializedError):
        runtime.play_animation(asset)
        
    with pytest.raises(RuntimeNotInitializedError):
        runtime.play_sequence([asset])
        
    with pytest.raises(RuntimeNotInitializedError):
        runtime.stop()


def test_runtime_invalid_state_transition():
    """Test invalid state handling."""
    runtime = MockAvatarRuntimeService()
    
    # Force error state to test constraint
    runtime._state = AvatarRuntimeState.ERROR
    
    with pytest.raises(InvalidRuntimeStateError):
        runtime.play_animation(create_mock_asset("HELLO"))
