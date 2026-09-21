"""Tests for the Phase 7.4 Avatar Controller orchestration."""

import pytest

from backend.schemas.sign_sequence import SignSequenceResult, SignSequenceItem
from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_animation_service import MockAnimationAssetService
from backend.services.avatar_runtime_service import MockAvatarRuntimeService, RuntimeNotInitializedError, AvatarRuntimeState
from backend.services.avatar_playback_service import SequentialPlaybackService, AvatarPlaybackError, PlaybackState
from backend.services.avatar_controller_service import AvatarControllerService


@pytest.fixture
def animation_service():
    service = MockAnimationAssetService()
    service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="test_hello_123", availability=AnimationAvailability.AVAILABLE))
    service.register_asset(AnimationAsset(sign_id="WORLD", asset_id="test_world_123", availability=AnimationAvailability.AVAILABLE))
    return service


@pytest.fixture
def runtime_service():
    service = MockAvatarRuntimeService()
    service.initialize()
    return service


@pytest.fixture
def playback_service(runtime_service):
    return SequentialPlaybackService(runtime_service)


@pytest.fixture
def controller(animation_service, playback_service):
    return AvatarControllerService(
        animation_service=animation_service,
        playback_service=playback_service
    )


def create_sequence(sign_ids, unsupported_tokens=None):
    items = []
    for i, sign_id in enumerate(sign_ids):
        items.append(SignSequenceItem(sign_id=sign_id, source_text="dummy", original_index=i))
    return SignSequenceResult(sequence=items, unsupported_tokens=unsupported_tokens or [])


def test_controller_basic_flow(controller, playback_service, runtime_service):
    """Test basic successful orchestration."""
    seq = create_sequence(["HELLO", "WORLD"])
    result = controller.play_sign_sequence(seq)
    
    assert result.accepted is True
    assert result.resolved_animation_count == 2
    assert result.unsupported_semantic_tokens == []
    
    # Verify playback service
    assert playback_service.get_state() == PlaybackState.PLAYING
    assert playback_service.get_current_index() == 0
    assert playback_service.get_current_asset().asset_id == "test_hello_123"
    
    # Verify runtime
    assert runtime_service.get_state() == AvatarRuntimeState.PLAYING
    history = runtime_service.get_history()
    assert len(history) == 1
    assert history[0].asset_id == "test_hello_123"


def test_controller_ordering_and_duplicates(controller, playback_service):
    """Test that controller preserves sequence order and duplicates exactly."""
    seq = create_sequence(["HELLO", "HELLO", "WORLD", "HELLO"])
    result = controller.play_sign_sequence(seq)
    
    assert result.accepted is True
    assert result.resolved_animation_count == 4
    
    assert playback_service.get_current_asset().sign_id == "HELLO"
    
    # Complete one to verify it moves forward
    playback_service.on_animation_complete()
    assert playback_service.get_current_asset().sign_id == "HELLO"


def test_controller_unsupported_tokens(controller, playback_service):
    """Test that unsupported semantic tokens are passed through to the result."""
    seq = create_sequence(["HELLO"], unsupported_tokens=["banana", "apple"])
    result = controller.play_sign_sequence(seq)
    
    assert result.accepted is True
    assert result.resolved_animation_count == 1
    assert result.unsupported_semantic_tokens == ["banana", "apple"]
    
    assert playback_service.get_state() == PlaybackState.PLAYING


def test_controller_unavailable_animation(controller, animation_service):
    """Test that unavailable animations correctly propagate playback error from runtime."""
    # Register an explicitly unavailable asset
    animation_service.register_asset(AnimationAsset(
        sign_id="MISSING", 
        asset_id="missing", 
        availability=AnimationAvailability.UNAVAILABLE
    ))
    
    seq = create_sequence(["HELLO", "MISSING"])
    
    with pytest.raises(AvatarPlaybackError):
        controller.play_sign_sequence(seq)


def test_controller_runtime_not_initialized(animation_service):
    """Test controller cleanly propagates uninitialized runtime error."""
    # Create an uninitialized runtime
    runtime = MockAvatarRuntimeService()
    playback = SequentialPlaybackService(runtime)
    
    ctrl = AvatarControllerService(
        animation_service=animation_service,
        playback_service=playback
    )
    
    seq = create_sequence(["HELLO"])
    
    with pytest.raises(RuntimeNotInitializedError):
        ctrl.play_sign_sequence(seq)


def test_controller_empty_sequence(controller, playback_service, runtime_service):
    """Test that empty sequences are handled gracefully as a no-op."""
    seq = create_sequence([])
    result = controller.play_sign_sequence(seq)
    
    assert result.accepted is True
    assert result.resolved_animation_count == 0
    
    assert playback_service.get_state() == PlaybackState.COMPLETED
    assert len(runtime_service.get_history()) == 0
