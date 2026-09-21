"""Tests for the Phase 7.1 Avatar Architecture."""

import pytest
from pydantic import ValidationError

from backend.schemas.avatar import AvatarSignCommand, AvatarSequence
from backend.services.avatar_service import MockAvatarService


def test_avatar_sign_command_validation():
    """Test AvatarSignCommand requires valid inputs."""
    # Valid
    command = AvatarSignCommand(sign_id="HELLO", source_text="hello", sequence_index=0)
    assert command.sign_id == "HELLO"
    
    # Missing sign_id
    with pytest.raises(ValidationError):
        AvatarSignCommand(source_text="hello", sequence_index=0)
        
    # Empty sign_id
    with pytest.raises(ValidationError):
        AvatarSignCommand(sign_id="", source_text="hello", sequence_index=0)


def test_avatar_sequence_ordering():
    """Test AvatarSequence strictly preserves order and duplicates."""
    cmd1 = AvatarSignCommand(sign_id="HELLO", sequence_index=0)
    cmd2 = AvatarSignCommand(sign_id="HELLO", sequence_index=1)
    cmd3 = AvatarSignCommand(sign_id="THANK_YOU", sequence_index=2)
    
    seq = AvatarSequence(sequence=[cmd1, cmd2, cmd3])
    
    assert len(seq.sequence) == 3
    assert seq.sequence[0].sign_id == "HELLO"
    assert seq.sequence[1].sign_id == "HELLO"
    assert seq.sequence[2].sign_id == "THANK_YOU"


def test_avatar_sequence_empty():
    """Test AvatarSequence handles empty sequences gracefully according to schema."""
    seq = AvatarSequence(sequence=[])
    assert len(seq.sequence) == 0


def test_mock_avatar_provider():
    """Test the MockAvatarService implementation."""
    provider = MockAvatarService()
    
    assert provider.get_current_command() is None
    
    cmd = AvatarSignCommand(sign_id="PLEASE", sequence_index=0)
    provider.play_sign(cmd)
    
    assert provider.get_current_command() == cmd
    
    provider.reset()
    assert provider.get_current_command() is None


def test_mock_avatar_provider_sequence():
    """Test MockAvatarService correctly processes sequences."""
    provider = MockAvatarService()
    
    seq = AvatarSequence(sequence=[
        AvatarSignCommand(sign_id="HELLO", sequence_index=0),
        AvatarSignCommand(sign_id="WORLD", sequence_index=1),
        AvatarSignCommand(sign_id="HELLO", sequence_index=2)
    ])
    
    provider.play_sequence(seq)
    
    history = provider.get_history()
    assert len(history) == 3
    
    assert history[0].sign_id == "HELLO"
    assert history[1].sign_id == "WORLD"
    assert history[2].sign_id == "HELLO"
    assert provider.get_current_command().sign_id == "HELLO"
