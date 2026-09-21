"""Tests for the Phase 7.2 Avatar Animation Contract."""

import pytest
from pydantic import ValidationError

from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_animation_service import MockAnimationAssetService


def test_animation_asset_validation():
    """Test AnimationAsset requires valid core inputs."""
    # Valid
    asset = AnimationAsset(sign_id="HELLO", asset_id="hello_default")
    assert asset.sign_id == "HELLO"
    assert asset.asset_id == "hello_default"
    assert asset.availability == AnimationAvailability.AVAILABLE
    assert asset.loop is False
    assert asset.speed == 1.0
    
    # Missing sign_id
    with pytest.raises(ValidationError):
        AnimationAsset(asset_id="hello_default")
        
    # Empty sign_id
    with pytest.raises(ValidationError):
        AnimationAsset(sign_id="", asset_id="hello_default")

    # Missing asset_id
    with pytest.raises(ValidationError):
        AnimationAsset(sign_id="HELLO")


def test_animation_asset_metadata():
    """Test optional playback metadata fields."""
    asset = AnimationAsset(
        sign_id="THANK_YOU",
        asset_id="thank_you_formal",
        duration=2.5,
        loop=True,
        speed=1.5,
        variant="formal"
    )
    
    assert asset.duration == 2.5
    assert asset.loop is True
    assert asset.speed == 1.5
    assert asset.variant == "formal"


def test_mock_resolver_known_sign():
    """Test the resolver returns registered assets correctly."""
    resolver = MockAnimationAssetService()
    
    # Register an asset
    asset = AnimationAsset(sign_id="HELLO", asset_id="hello_default")
    resolver.register_asset(asset)
    
    # Resolve it
    resolved = resolver.resolve_sign("HELLO")
    assert resolved.sign_id == "HELLO"
    assert resolved.asset_id == "hello_default"
    assert resolved.availability == AnimationAvailability.AVAILABLE
    
    # Mutating the resolved asset should not mutate the registered mapping
    resolved.speed = 2.0
    resolved2 = resolver.resolve_sign("HELLO")
    assert resolved2.speed == 1.0


def test_mock_resolver_unknown_sign():
    """Test the resolver returns an unavailable status for unknown signs without crashing."""
    resolver = MockAnimationAssetService()
    
    resolved = resolver.resolve_sign("UNKNOWN_SIGN")
    assert resolved.sign_id == "UNKNOWN_SIGN"
    assert resolved.availability == AnimationAvailability.UNAVAILABLE


def test_mock_resolver_sequence():
    """Test the resolver strictly preserves order and handles duplicates/unknowns."""
    resolver = MockAnimationAssetService()
    
    resolver.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_default"))
    resolver.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="thank_you_default"))
    
    sequence = ["HELLO", "UNKNOWN_SIGN", "HELLO", "THANK_YOU"]
    resolved_seq = resolver.resolve_sequence(sequence)
    
    assert len(resolved_seq) == 4
    assert resolved_seq[0].sign_id == "HELLO"
    assert resolved_seq[0].availability == AnimationAvailability.AVAILABLE
    
    assert resolved_seq[1].sign_id == "UNKNOWN_SIGN"
    assert resolved_seq[1].availability == AnimationAvailability.UNAVAILABLE
    
    assert resolved_seq[2].sign_id == "HELLO"
    assert resolved_seq[2].availability == AnimationAvailability.AVAILABLE
    
    assert resolved_seq[3].sign_id == "THANK_YOU"
    assert resolved_seq[3].availability == AnimationAvailability.AVAILABLE
