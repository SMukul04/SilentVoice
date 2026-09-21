"""Provider-independent runtime architecture for 3D avatar rendering."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability


class AvatarRuntimeState(str, Enum):
    """Represents the lifecycle state of the Avatar renderer."""
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    PLAYING = "PLAYING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ==============================================================================
# Exceptions
# ==============================================================================

class AvatarRuntimeError(Exception):
    """Base exception for Avatar Runtime errors."""
    pass


class RuntimeNotInitializedError(AvatarRuntimeError):
    """Raised when an operation requires the runtime to be READY, but it is UNINITIALIZED."""
    pass


class InvalidRuntimeStateError(AvatarRuntimeError):
    """Raised when a state transition or operation is invalid for the current state."""
    pass


class AnimationPlaybackError(AvatarRuntimeError):
    """Raised when an animation asset cannot be played (e.g. unavailable)."""
    pass


# ==============================================================================
# Interface
# ==============================================================================

class AvatarRuntimeService(ABC):
    """Abstract boundary for an Avatar rendering engine (e.g., Unity)."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the runtime, preparing it for playback."""
        pass

    @abstractmethod
    def play_animation(self, asset: AnimationAsset) -> None:
        """Command the runtime to play a specific animation asset."""
        pass

    @abstractmethod
    def play_sequence(self, assets: List[AnimationAsset]) -> None:
        """Command the runtime to play a sequence of animation assets."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop current playback."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the runtime state completely, clearing playback state."""
        pass

    @abstractmethod
    def get_state(self) -> AvatarRuntimeState:
        """Get the current state of the runtime."""
        pass
        
    @abstractmethod
    def get_current_animation(self) -> Optional[AnimationAsset]:
        """Get the currently playing animation asset, if any."""
        pass


# ==============================================================================
# Mock Implementation
# ==============================================================================

class MockAvatarRuntimeService(AvatarRuntimeService):
    """In-memory runtime mock for testing backend behavior without a 3D renderer."""
    
    def __init__(self):
        self._state: AvatarRuntimeState = AvatarRuntimeState.UNINITIALIZED
        self._current_animation: Optional[AnimationAsset] = None
        self._playback_history: List[AnimationAsset] = []

    def initialize(self) -> None:
        if self._state == AvatarRuntimeState.ERROR:
            # Re-initialize clears errors
            pass
        self._state = AvatarRuntimeState.READY
        self._current_animation = None

    def play_animation(self, asset: AnimationAsset) -> None:
        if self._state == AvatarRuntimeState.UNINITIALIZED:
            raise RuntimeNotInitializedError("Cannot play animation: runtime is not initialized.")
            
        if self._state == AvatarRuntimeState.ERROR:
            raise InvalidRuntimeStateError("Cannot play animation: runtime is in an ERROR state.")
            
        if asset.availability == AnimationAvailability.UNAVAILABLE:
            raise AnimationPlaybackError(f"Cannot play unavailable asset for sign: {asset.sign_id}")
            
        self._state = AvatarRuntimeState.PLAYING
        self._current_animation = asset
        self._playback_history.append(asset)
        
        # State remains PLAYING until a real provider finishes.

    def play_sequence(self, assets: List[AnimationAsset]) -> None:
        if self._state == AvatarRuntimeState.UNINITIALIZED:
            raise RuntimeNotInitializedError("Cannot play sequence: runtime is not initialized.")
            
        if self._state == AvatarRuntimeState.ERROR:
            raise InvalidRuntimeStateError("Cannot play sequence: runtime is in an ERROR state.")
            
        if not assets:
            # no-op for empty sequences
            return
            
        # Validate all assets before playing
        for asset in assets:
            if asset.availability == AnimationAvailability.UNAVAILABLE:
                raise AnimationPlaybackError(f"Cannot play sequence with unavailable asset: {asset.sign_id}")
            
        self._state = AvatarRuntimeState.PLAYING
        for asset in assets:
            self._current_animation = asset
            self._playback_history.append(asset)
            
        # State remains PLAYING until a real provider finishes.

    def stop(self) -> None:
        if self._state == AvatarRuntimeState.UNINITIALIZED:
            raise RuntimeNotInitializedError("Cannot stop: runtime is not initialized.")
            
        self._state = AvatarRuntimeState.STOPPED
        self._current_animation = None

    def reset(self) -> None:
        """Resets to UNINITIALIZED and clears history."""
        self._state = AvatarRuntimeState.UNINITIALIZED
        self._current_animation = None
        self._playback_history.clear()

    def get_state(self) -> AvatarRuntimeState:
        return self._state
        
    def get_current_animation(self) -> Optional[AnimationAsset]:
        return self._current_animation
        
    # Helper for testing
    def get_history(self) -> List[AnimationAsset]:
        return list(self._playback_history)
