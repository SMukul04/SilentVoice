"""Module defining the FrameFeatures dataclass to encapsulate features of hands detected in a single frame."""

from dataclasses import dataclass
from typing import Optional
from backend.sign_recognition.hand_features import HandFeatures


@dataclass
class FrameFeatures:
    """Encapsulates hand landmark features detected in a single video frame.

    This class serves as a container for the features of both the left and right
    hands detected in a frame, facilitating the transition from single-hand to
    two-hand gesture recognition.

    Attributes:
        left_hand (Optional[HandFeatures]): The features for the detected left hand,
            or None if no left hand was detected.
        right_hand (Optional[HandFeatures]): The features for the detected right hand,
            or None if no right hand was detected.
    """
    left_hand: Optional[HandFeatures] = None
    right_hand: Optional[HandFeatures] = None

    def __post_init__(self) -> None:
        """Validates the input parameters after initialization.

        Raises:
            TypeError: If left_hand/right_hand is not an instance of HandFeatures (when present).
            ValueError: If the handedness attribute does not match the respective hand field.
        """
        if self.left_hand is not None:
            if not isinstance(self.left_hand, HandFeatures):
                raise TypeError(
                    f"left_hand must be an instance of HandFeatures, got {type(self.left_hand)}"
                )
            if self.left_hand.handedness != "Left":
                raise ValueError(
                    f"left_hand must have handedness 'Left', got '{self.left_hand.handedness}'"
                )

        if self.right_hand is not None:
            if not isinstance(self.right_hand, HandFeatures):
                raise TypeError(
                    f"right_hand must be an instance of HandFeatures, got {type(self.right_hand)}"
                )
            if self.right_hand.handedness != "Right":
                raise ValueError(
                    f"right_hand must have handedness 'Right', got '{self.right_hand.handedness}'"
                )

    def has_left_hand(self) -> bool:
        """Checks if a left hand is present in the frame.

        Returns:
            bool: True if left_hand is not None, False otherwise.
        """
        return self.left_hand is not None

    def has_right_hand(self) -> bool:
        """Checks if a right hand is present in the frame.

        Returns:
            bool: True if right_hand is not None, False otherwise.
        """
        return self.right_hand is not None

    def has_both_hands(self) -> bool:
        """Checks if both left and right hands are present in the frame.

        Returns:
            bool: True if both left_hand and right_hand are not None, False otherwise.
        """
        return self.left_hand is not None and self.right_hand is not None

    def has_any_hand(self) -> bool:
        """Checks if at least one hand (left or right) is present in the frame.

        Returns:
            bool: True if either left_hand or right_hand is not None, False otherwise.
        """
        return self.left_hand is not None or self.right_hand is not None
