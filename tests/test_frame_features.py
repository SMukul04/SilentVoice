import unittest
import numpy as np
from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_features import FrameFeatures


class TestFrameFeatures(unittest.TestCase):
    """Unit tests for the FrameFeatures dataclass."""

    def setUp(self) -> None:
        """Sets up mock HandFeatures for testing."""
        # Create standard left and right HandFeatures objects
        landmarks = np.zeros(63, dtype=np.float32)
        self.left_hand = HandFeatures(landmarks=landmarks, handedness="Left", confidence=0.9)
        self.right_hand = HandFeatures(landmarks=landmarks, handedness="Right", confidence=0.85)

    def test_default_initialization(self) -> None:
        """Tests that FrameFeatures defaults to None for both hands."""
        frame = FrameFeatures()
        self.assertIsNone(frame.left_hand)
        self.assertIsNone(frame.right_hand)
        self.assertFalse(frame.has_left_hand())
        self.assertFalse(frame.has_right_hand())
        self.assertFalse(frame.has_both_hands())
        self.assertFalse(frame.has_any_hand())

    def test_one_hand_initialization(self) -> None:
        """Tests FrameFeatures with only one hand present."""
        # Left hand only
        frame_left = FrameFeatures(left_hand=self.left_hand)
        self.assertTrue(frame_left.has_left_hand())
        self.assertFalse(frame_left.has_right_hand())
        self.assertFalse(frame_left.has_both_hands())
        self.assertTrue(frame_left.has_any_hand())

        # Right hand only
        frame_right = FrameFeatures(right_hand=self.right_hand)
        self.assertFalse(frame_right.has_left_hand())
        self.assertTrue(frame_right.has_right_hand())
        self.assertFalse(frame_right.has_both_hands())
        self.assertTrue(frame_right.has_any_hand())

    def test_both_hands_initialization(self) -> None:
        """Tests FrameFeatures with both hands present."""
        frame = FrameFeatures(left_hand=self.left_hand, right_hand=self.right_hand)
        self.assertTrue(frame.has_left_hand())
        self.assertTrue(frame.has_right_hand())
        self.assertTrue(frame.has_both_hands())
        self.assertTrue(frame.has_any_hand())

    def test_validation_type_errors(self) -> None:
        """Tests that passing invalid types raises TypeError."""
        with self.assertRaises(TypeError):
            FrameFeatures(left_hand="not_a_hand")  # type: ignore

        with self.assertRaises(TypeError):
            FrameFeatures(right_hand=123)  # type: ignore

    def test_validation_value_errors(self) -> None:
        """Tests that passing a hand with mismatched handedness raises ValueError."""
        # Passing a hand labeled "Right" to left_hand
        with self.assertRaises(ValueError):
            FrameFeatures(left_hand=self.right_hand)

        # Passing a hand labeled "Left" to right_hand
        with self.assertRaises(ValueError):
            FrameFeatures(right_hand=self.left_hand)
