"""Provider-independent interface for resolving semantic signs to visual animation assets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability


class AnimationAssetService(ABC):
    """Abstract interface for resolving animation assets from semantic signs."""

    @abstractmethod
    def resolve_sign(self, sign_id: str, variant: Optional[str] = None) -> AnimationAsset:
        """Resolve a single semantic sign_id to its corresponding AnimationAsset.
        
        Must never throw an exception for an unknown sign. It should return an 
        AnimationAsset with availability=UNAVAILABLE instead.
        """
        pass

    @abstractmethod
    def resolve_sequence(self, sign_ids: List[str]) -> List[AnimationAsset]:
        """Resolve a sequence of sign_ids to their AnimationAssets, preserving order."""
        pass


class MockAnimationAssetService(AnimationAssetService):
    """In-memory resolver for testing without a real animation database."""
    
    def __init__(self, mappings: Optional[Dict[str, AnimationAsset]] = None):
        self._mappings = mappings or {}

    def resolve_sign(self, sign_id: str, variant: Optional[str] = None) -> AnimationAsset:
        # Check if we have an explicit mapping
        asset = self._mappings.get(sign_id)
        if asset:
            # We return a copy to prevent mutation of the registered mapping
            return asset.model_copy()
            
        # Return unavailable asset
        return AnimationAsset(
            sign_id=sign_id,
            asset_id=f"unknown_{sign_id.lower()}",
            availability=AnimationAvailability.UNAVAILABLE
        )

    def resolve_sequence(self, sign_ids: List[str]) -> List[AnimationAsset]:
        return [self.resolve_sign(sign_id) for sign_id in sign_ids]
    
    def register_asset(self, asset: AnimationAsset) -> None:
        """Helper to inject assets for testing."""
        self._mappings[asset.sign_id] = asset.model_copy()
