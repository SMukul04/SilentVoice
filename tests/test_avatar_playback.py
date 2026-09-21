"""Tests for Phase 7.5 Avatar Playback Service."""

import pytest

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_runtime_service import MockAvatarRuntimeService, AvatarRuntimeState, RuntimeNotInitializedError
from backend.services.avatar_playback_service import SequentialPlaybackService, PlaybackState, AvatarPlaybackError


def create_asset(sign_id: str, available=True) -> AnimationAsset:
    return AnimationAsset(
        sign_id=sign_id,
        asset_id=f"test_{sign_id.lower()}",
        availability=AnimationAvailability.AVAILABLE if available else AnimationAvailability.UNAVAILABLE
    )


def test_playback_start_flow():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A"), create_asset("B"), create_asset("C")]
    playback.play_sequence(assets)
    
    assert playback.get_state() == PlaybackState.PLAYING
    assert playback.get_current_index() == 0
    assert playback.get_current_asset() == assets[0]
    
    assert runtime.get_state() == AvatarRuntimeState.PLAYING
    assert runtime.get_current_animation() == assets[0]


def test_playback_completion_flow():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A"), create_asset("B"), create_asset("C")]
    playback.play_sequence(assets)
    
    # Complete A
    playback.on_animation_complete()
    assert playback.get_current_index() == 1
    assert runtime.get_current_animation() == assets[1]
    
    # Complete B
    playback.on_animation_complete()
    assert playback.get_current_index() == 2
    assert runtime.get_current_animation() == assets[2]
    
    # Complete C
    playback.on_animation_complete()
    assert playback.get_state() == PlaybackState.COMPLETED
    assert playback.get_current_index() is None


def test_playback_ordering_and_duplicates():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A"), create_asset("A"), create_asset("B")]
    playback.play_sequence(assets)
    
    assert playback.get_current_asset().sign_id == "A"
    
    playback.on_animation_complete()
    assert playback.get_current_asset().sign_id == "A"
    
    playback.on_animation_complete()
    assert playback.get_current_asset().sign_id == "B"
    
    playback.on_animation_complete()
    assert playback.get_state() == PlaybackState.COMPLETED
    
    # Verify exact runtime hits
    history = runtime.get_history()
    assert len(history) == 3
    assert history[0].sign_id == "A"
    assert history[1].sign_id == "A"
    assert history[2].sign_id == "B"


def test_playback_empty_sequence():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    playback.play_sequence([])
    
    assert playback.get_state() == PlaybackState.COMPLETED
    assert playback.get_current_index() is None
    assert len(runtime.get_history()) == 0


def test_playback_unavailable_asset():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A"), create_asset("B", available=False), create_asset("C")]
    
    with pytest.raises(AvatarPlaybackError):
        playback.play_sequence(assets)
        
    # Runtime should not have been called for A
    assert len(runtime.get_history()) == 0


def test_playback_runtime_not_initialized():
    runtime = MockAvatarRuntimeService()
    # Purposely do not call initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A")]
    
    with pytest.raises(RuntimeNotInitializedError):
        playback.play_sequence(assets)


def test_playback_runtime_failure_propagation():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    # Force the mock into ERROR state so the next play fails
    runtime._state = AvatarRuntimeState.ERROR
    
    playback = SequentialPlaybackService(runtime)
    assets = [create_asset("A"), create_asset("B")]
    
    with pytest.raises(Exception):
        playback.play_sequence(assets)
        
    assert playback.get_state() == PlaybackState.ERROR
    assert len(runtime.get_history()) == 0


def test_playback_stop_behavior():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    assets = [create_asset("A"), create_asset("B")]
    playback.play_sequence(assets)
    
    playback.stop()
    assert playback.get_state() == PlaybackState.STOPPED
    assert playback.get_current_index() is None
    
    # Runtime should be stopped
    assert runtime.get_state() == AvatarRuntimeState.STOPPED
    assert len(runtime.get_history()) == 1  # Only A was requested


def test_playback_reset_behavior():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    
    playback.play_sequence([create_asset("A")])
    playback.reset()
    
    assert playback.get_state() == PlaybackState.IDLE
    assert playback.get_current_index() is None
    assert runtime.get_state() == AvatarRuntimeState.UNINITIALIZED
