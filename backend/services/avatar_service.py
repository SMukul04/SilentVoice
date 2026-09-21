"""Provider-independent interface for Avatar operations.

This module defines the architectural boundary between the backend semantic
sign sequencing and the frontend/Unity visual 3D avatar implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from backend.schemas.avatar import AvatarSignCommand, AvatarSequence


class AvatarService(ABC):
    """Abstract interface for an Avatar provider.
    
    This ensures the backend is decoupled from the specific rendering
    technology (e.g., Unity WebGL, Three.js, etc.).
    """

    @abstractmethod
    def play_sign(self, command: AvatarSignCommand) -> None:
        """Command the avatar to play a single sign."""
        pass

    @abstractmethod
    def play_sequence(self, sequence: AvatarSequence) -> None:
        """Command the avatar to play a sequence of signs in order."""
        pass

    @abstractmethod
    def get_current_command(self) -> Optional[AvatarSignCommand]:
        """Get the last played command or current state."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the avatar to its idle or default state."""
        pass


class MockAvatarService(AvatarService):
    """A lightweight mock provider for backend testing and validation.
    
    Maintains an in-memory history of commands and guarantees sequence order
    without requiring a 3D rendering engine.
    """
    
    def __init__(self):
        self._current_command: Optional[AvatarSignCommand] = None
        self._command_history: List[AvatarSignCommand] = []

    def play_sign(self, command: AvatarSignCommand) -> None:
        self._current_command = command
        self._command_history.append(command)

    def play_sequence(self, sequence: AvatarSequence) -> None:
        for command in sequence.sequence:
            self.play_sign(command)

    def get_current_command(self) -> Optional[AvatarSignCommand]:
        return self._current_command

    def reset(self) -> None:
        self._current_command = None
        self._command_history.clear()
        
    def get_history(self) -> List[AvatarSignCommand]:
        """Expose command history for testing."""
        return list(self._command_history)
