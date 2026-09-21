"""Avatar Controller Service for orchestrating sign sequences to runtime."""

from typing import List, Optional
from pydantic import BaseModel, Field

from backend.schemas.avatar_animation import AnimationAsset
from backend.schemas.sign_sequence import SignSequenceResult
from backend.services.avatar_animation_service import AnimationAssetService
from backend.services.avatar_playback_service import SequentialPlaybackService, PlaybackState
from backend.services.avatar_runtime_service import AvatarRuntimeError


class AvatarControlError(Exception):
    """Base exception for avatar control logic errors."""
    pass


class PlaybackAlreadyActiveError(AvatarControlError):
    """Raised when attempting to play a sequence while one is already playing."""
    pass


class InvalidPlaybackControlError(AvatarControlError):
    """Raised when attempting a control operation in an invalid state."""
    pass


class AvatarControllerResult(BaseModel):
    """Result of playing a sign sequence through the avatar controller."""
    accepted: bool = Field(..., description="Whether the sequence was accepted for playback.")
    resolved_animation_count: int = Field(..., description="Number of animation assets resolved and sent to runtime.")
    unsupported_semantic_tokens: List[str] = Field(..., description="Tokens that failed semantic sign resolution.")
    error_message: str = Field(None, description="Error message if playback failed.")


class AvatarPlaybackProgress(BaseModel):
    """Represents the current progress of sequence playback."""
    state: PlaybackState = Field(..., description="Current playback state.")
    current_index: Optional[int] = Field(..., description="The current animation index being played.")
    total_assets: int = Field(..., description="Total number of animations in the current sequence.")
    current_asset: Optional[AnimationAsset] = Field(..., description="The currently playing animation asset, if any.")


class AvatarControllerService:
    """Orchestrates the resolution of SignSequences to AnimationAssets and their playback on the AvatarRuntime."""

    def __init__(
        self,
        animation_service: AnimationAssetService,
        playback_service: SequentialPlaybackService
    ):
        self._animation_service = animation_service
        self._playback_service = playback_service

    def play_sign_sequence(self, sign_sequence: SignSequenceResult) -> AvatarControllerResult:
        """
        Coordinates playing a sign sequence.
        
        1. Validates current state.
        2. Extracts semantic sign IDs.
        3. Resolves them to AnimationAssets.
        4. Issues playback to the SequentialPlaybackService.
        """
        current_state = self._playback_service.get_state()
        
        if current_state == PlaybackState.PLAYING:
            raise PlaybackAlreadyActiveError("Cannot play a new sequence while playback is active.")
            
        if current_state == PlaybackState.ERROR:
            raise InvalidPlaybackControlError("Playback is in an ERROR state. You must call reset() first.")
            
        if not sign_sequence.sequence:
            self._playback_service.play_sequence([])
            return AvatarControllerResult(
                accepted=True,
                resolved_animation_count=0,
                unsupported_semantic_tokens=sign_sequence.unsupported_tokens
            )
            
        sign_ids = [item.sign_id for item in sign_sequence.sequence]
        
        # Use existing AnimationAssetService to resolve the sequence
        assets = self._animation_service.resolve_sequence(sign_ids)
        
        # The SequentialPlaybackService will natively handle 
        # raising exceptions for unavailable assets and uninitialized runtime.
        try:
            self._playback_service.play_sequence(assets)
        except Exception as e:
            # We catch exceptions to return a failure if needed, but per previous phase we propagated them.
            # Let's preserve propagation behavior as requested, but we can also just raise it.
            raise
        
        return AvatarControllerResult(
            accepted=True,
            resolved_animation_count=len(assets),
            unsupported_semantic_tokens=sign_sequence.unsupported_tokens
        )

    def stop(self) -> None:
        """Stop current sequence playback."""
        # Stop is safe to call at any time; playback service handles state transition appropriately.
        self._playback_service.stop()

    def reset(self) -> None:
        """Reset the avatar system to a clean state."""
        self._playback_service.reset()

    def get_state(self) -> PlaybackState:
        """Get the current playback state."""
        return self._playback_service.get_state()
        
    def get_progress(self) -> AvatarPlaybackProgress:
        """Get structured progress of the current sequence playback."""
        return AvatarPlaybackProgress(
            state=self._playback_service.get_state(),
            current_index=self._playback_service.get_current_index(),
            total_assets=self._playback_service.get_total_assets(),
            current_asset=self._playback_service.get_current_asset()
        )
        
    def get_current_animation(self) -> Optional[AnimationAsset]:
        """Get the animation currently playing, if any."""
        return self._playback_service.get_current_asset()
