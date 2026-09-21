"""Tests for Phase 7.6 Avatar Controls / Lifecycle."""

import pytest

from backend.schemas.sign_sequence import SignSequenceResult, SignSequenceItem
from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_animation_service import MockAnimationAssetService
from backend.services.avatar_runtime_service import MockAvatarRuntimeService, AvatarRuntimeState, AvatarRuntimeError
from backend.services.avatar_playback_service import SequentialPlaybackService, PlaybackState
from backend.services.avatar_controller_service import (
    AvatarControllerService,
    PlaybackAlreadyActiveError,
    InvalidPlaybackControlError
)


@pytest.fixture
def animation_service():
    service = MockAnimationAssetService()
    service.register_asset(AnimationAsset(sign_id="A", asset_id="a_anim", availability=AnimationAvailability.AVAILABLE))
    service.register_asset(AnimationAsset(sign_id="B", asset_id="b_anim", availability=AnimationAvailability.AVAILABLE))
    service.register_asset(AnimationAsset(sign_id="C", asset_id="c_anim", availability=AnimationAvailability.AVAILABLE))
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


def create_seq(*sign_ids):
    items = [SignSequenceItem(sign_id=s, source_text="txt", original_index=i) for i, s in enumerate(sign_ids)]
    return SignSequenceResult(sequence=items, unsupported_tokens=[])


def test_repeated_play_rejected(controller):
    """Test playing a new sequence is rejected while another is active."""
    controller.play_sign_sequence(create_seq("A", "B"))
    
    assert controller.get_state() == PlaybackState.PLAYING
    
    with pytest.raises(PlaybackAlreadyActiveError):
        controller.play_sign_sequence(create_seq("C"))


def test_stop_behavior(controller, playback_service, runtime_service):
    """Test stopping sequence playback correctly halts progression."""
    controller.play_sign_sequence(create_seq("A", "B", "C"))
    assert controller.get_state() == PlaybackState.PLAYING
    
    controller.stop()
    assert controller.get_state() == PlaybackState.STOPPED
    assert controller.get_current_animation() is None
    
    # Verify no progression
    playback_service.on_animation_complete()
    assert controller.get_state() == PlaybackState.STOPPED
    assert len(runtime_service.get_history()) == 1


def test_reset_behavior(controller, runtime_service):
    """Test resetting returns strictly to clean state."""
    controller.play_sign_sequence(create_seq("A", "B"))
    assert controller.get_state() == PlaybackState.PLAYING
    
    controller.reset()
    assert controller.get_state() == PlaybackState.IDLE
    assert runtime_service.get_state() == AvatarRuntimeState.UNINITIALIZED
    assert controller.get_current_animation() is None
    
    # Verify we can play again
    runtime_service.initialize()
    controller.play_sign_sequence(create_seq("C"))
    assert controller.get_state() == PlaybackState.PLAYING


def test_progress_semantics(controller, playback_service):
    """Test structured progress updates during completion signals."""
    controller.play_sign_sequence(create_seq("A", "B", "C"))
    
    prog1 = controller.get_progress()
    assert prog1.state == PlaybackState.PLAYING
    assert prog1.current_index == 0
    assert prog1.total_assets == 3
    assert prog1.current_asset.sign_id == "A"
    
    playback_service.on_animation_complete()
    
    prog2 = controller.get_progress()
    assert prog2.state == PlaybackState.PLAYING
    assert prog2.current_index == 1
    assert prog2.total_assets == 3
    assert prog2.current_asset.sign_id == "B"
    
    playback_service.on_animation_complete()
    playback_service.on_animation_complete()
    
    prog_done = controller.get_progress()
    assert prog_done.state == PlaybackState.COMPLETED
    assert prog_done.current_index is None
    assert prog_done.current_asset is None


def test_error_state_requires_reset(controller, runtime_service):
    """Test that a runtime failure pushes state to ERROR and requires reset."""
    runtime_service._state = AvatarRuntimeState.ERROR  # Mock an internal error state so play_animation throws
    
    with pytest.raises(AvatarRuntimeError):
        controller.play_sign_sequence(create_seq("A"))
        
    assert controller.get_state() == PlaybackState.ERROR
    
    with pytest.raises(InvalidPlaybackControlError):
        controller.play_sign_sequence(create_seq("B"))
        
    controller.reset()
    assert controller.get_state() == PlaybackState.IDLE
