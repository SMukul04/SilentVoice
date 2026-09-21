"""Sequential playback orchestration for avatar animations."""

from enum import Enum
from typing import List, Optional

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_runtime_service import (
    AvatarRuntimeService,
    RuntimeNotInitializedError,
    AvatarRuntimeError
)


class PlaybackState(str, Enum):
    """Lifecycle states of the sequential playback session."""
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AvatarPlaybackError(Exception):
    """Base error for playback service issues."""
    pass


class SequentialPlaybackService:
    """Manages the deterministic sequencing of animation playback."""

    def __init__(self, runtime_service: AvatarRuntimeService):
        self._runtime = runtime_service
        self._state: PlaybackState = PlaybackState.IDLE
        self._sequence: List[AnimationAsset] = []
        self._current_index: Optional[int] = None
        self._error_message: Optional[str] = None

    def play_sequence(self, assets: List[AnimationAsset]) -> None:
        """Start playing a sequence of animation assets."""
        if not assets:
            self._state = PlaybackState.COMPLETED
            self._sequence = []
            self._current_index = None
            return

        # Pre-validate availability
        for asset in assets:
            if asset.availability == AnimationAvailability.UNAVAILABLE:
                raise AvatarPlaybackError(f"Cannot play sequence with unavailable asset: {asset.sign_id}")

        self._sequence = list(assets)
        self._current_index = 0
        self._state = PlaybackState.PLAYING
        self._error_message = None

        self._play_current()

    def on_animation_complete(self) -> None:
        """Signal from the runtime provider that the current animation has finished."""
        if self._state != PlaybackState.PLAYING:
            return
            
        if self._current_index is None:
            return

        self._current_index += 1

        if self._current_index >= len(self._sequence):
            self._state = PlaybackState.COMPLETED
            self._current_index = None
        else:
            self._play_current()

    def _play_current(self) -> None:
        """Helper to invoke the runtime for the current asset."""
        if self._current_index is None or self._current_index >= len(self._sequence):
            return
            
        asset = self._sequence[self._current_index]
        try:
            self._runtime.play_animation(asset)
        except AvatarRuntimeError as e:
            self._state = PlaybackState.ERROR
            self._error_message = str(e)
            # We explicitly raise runtime errors back up on the start call,
            # or they are captured in state if raised during on_animation_complete
            raise

    def stop(self) -> None:
        """Halt sequence playback."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.STOPPED
            self._current_index = None
            self._runtime.stop()

    def reset(self) -> None:
        """Clear playback sequence and restore IDLE state."""
        self._state = PlaybackState.IDLE
        self._sequence = []
        self._current_index = None
        self._error_message = None
        self._runtime.reset()

    # Accessors
    def get_state(self) -> PlaybackState:
        return self._state

    def get_current_index(self) -> Optional[int]:
        return self._current_index

    def get_current_asset(self) -> Optional[AnimationAsset]:
        if self._current_index is not None and self._current_index < len(self._sequence):
            return self._sequence[self._current_index]
        return None

    def get_total_assets(self) -> int:
        return len(self._sequence)
        
    def get_error_message(self) -> Optional[str]:
        return self._error_message
