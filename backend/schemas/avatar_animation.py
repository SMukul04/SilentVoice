"""Data contracts for the Avatar Animation Asset mapping."""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class AnimationAvailability(str, Enum):
    """Indicates whether an animation asset is available for a sign."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class AnimationAsset(BaseModel):
    """Represents a provider-independent visual animation asset."""
    sign_id: str = Field(..., min_length=1, description="The core semantic identity of the sign.")
    asset_id: str = Field(..., min_length=1, description="The provider-neutral identifier for the animation resource.")
    availability: AnimationAvailability = Field(
        default=AnimationAvailability.AVAILABLE,
        description="Whether this asset is actually available for playback."
    )
    
    # Playback Metadata
    duration: Optional[float] = Field(None, description="Expected duration in seconds, if known.")
    loop: bool = Field(False, description="Whether the animation should loop.")
    speed: float = Field(1.0, description="Playback speed multiplier.")
    variant: Optional[str] = Field(None, description="Optional variant of the sign (e.g., 'formal', 'casual').")
