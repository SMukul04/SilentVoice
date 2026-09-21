"""Avatar Controller Service for orchestrating sign sequences to runtime."""

from typing import List
from pydantic import BaseModel, Field

from backend.schemas.sign_sequence import SignSequenceResult
from backend.services.avatar_animation_service import AnimationAssetService
from backend.services.avatar_runtime_service import AvatarRuntimeService


class AvatarControllerResult(BaseModel):
    """Result of playing a sign sequence through the avatar controller."""
    accepted: bool = Field(..., description="Whether the sequence was accepted for playback.")
    resolved_animation_count: int = Field(..., description="Number of animation assets resolved and sent to runtime.")
    unsupported_semantic_tokens: List[str] = Field(..., description="Tokens that failed semantic sign resolution.")


class AvatarControllerService:
    """Orchestrates the resolution of SignSequences to AnimationAssets and their playback on the AvatarRuntime."""

    def __init__(
        self,
        animation_service: AnimationAssetService,
        runtime_service: AvatarRuntimeService
    ):
        self._animation_service = animation_service
        self._runtime_service = runtime_service

    def play_sign_sequence(self, sign_sequence: SignSequenceResult) -> AvatarControllerResult:
        """
        Coordinates playing a sign sequence.
        
        1. Extracts semantic sign IDs.
        2. Resolves them to AnimationAssets.
        3. Issues playback to the AvatarRuntime.
        """
        if not sign_sequence.sequence:
            return AvatarControllerResult(
                accepted=True,
                resolved_animation_count=0,
                unsupported_semantic_tokens=sign_sequence.unsupported_tokens
            )
            
        sign_ids = [item.sign_id for item in sign_sequence.sequence]
        
        # Use existing AnimationAssetService to resolve the sequence
        assets = self._animation_service.resolve_sequence(sign_ids)
        
        # The AvatarRuntimeService.play_sequence will natively handle 
        # raising AnimationPlaybackError for unavailable assets and
        # RuntimeNotInitializedError if not initialized.
        # We allow these exceptions to surface cleanly.
        self._runtime_service.play_sequence(assets)
        
        return AvatarControllerResult(
            accepted=True,
            resolved_animation_count=len(assets),
            unsupported_semantic_tokens=sign_sequence.unsupported_tokens
        )
