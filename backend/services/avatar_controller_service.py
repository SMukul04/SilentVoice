"""Avatar Controller Service for orchestrating sign sequences to runtime."""

from typing import List
from pydantic import BaseModel, Field

from backend.schemas.sign_sequence import SignSequenceResult
from backend.services.avatar_animation_service import AnimationAssetService
from backend.services.avatar_playback_service import SequentialPlaybackService
from backend.services.avatar_runtime_service import AvatarRuntimeError


class AvatarControllerResult(BaseModel):
    """Result of playing a sign sequence through the avatar controller."""
    accepted: bool = Field(..., description="Whether the sequence was accepted for playback.")
    resolved_animation_count: int = Field(..., description="Number of animation assets resolved and sent to runtime.")
    unsupported_semantic_tokens: List[str] = Field(..., description="Tokens that failed semantic sign resolution.")
    error_message: str = Field(None, description="Error message if playback failed.")


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
        
        1. Extracts semantic sign IDs.
        2. Resolves them to AnimationAssets.
        3. Issues playback to the SequentialPlaybackService.
        """
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
