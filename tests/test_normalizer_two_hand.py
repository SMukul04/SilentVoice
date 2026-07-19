import unittest
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_features import FrameFeatures
from backend.sign_recognition.normalizer import LandmarkNormalizer


class TestLandmarkNormalizerTwoHand(unittest.TestCase):
    """Tests the LandmarkNormalizer with different hand configurations (no hands, left only, right only, both)."""

    def setUp(self) -> None:
        self.normalizer = LandmarkNormalizer()
        
        # Setup mock hand landmarks
        left_coords = np.zeros((21, 3), dtype=np.float32)
        left_coords[0] = [1.0, 1.0, 1.0] # wrist
        left_coords[9] = [1.0, 3.0, 1.0] # middle MCP
        
        right_coords = np.zeros((21, 3), dtype=np.float32)
        right_coords[0] = [2.0, 2.0, 2.0] # wrist
        right_coords[9] = [2.0, 6.0, 2.0] # middle MCP

        self.left_hand = HandFeatures(
            landmarks=left_coords.flatten(),
            handedness="Left",
            confidence=0.9
        )
        self.right_hand = HandFeatures(
            landmarks=right_coords.flatten(),
            handedness="Right",
            confidence=0.95
        )

    def test_no_hands(self) -> None:
        """Verifies that no hands result in 126 zeros."""
        frame = FrameFeatures(left_hand=None, right_hand=None)
        out = self.normalizer.normalize(frame)
        self.assertEqual(out.shape, (126,))
        np.testing.assert_array_equal(out, np.zeros(126, dtype=np.float32))

    def test_left_hand_only(self) -> None:
        """Verifies that left hand only populates the first 63 values and leaves the rest as zeros."""
        frame = FrameFeatures(left_hand=self.left_hand, right_hand=None)
        out = self.normalizer.normalize(frame)
        self.assertEqual(out.shape, (126,))
        
        # First 63 values should not be all zeros (since wrist -> MCP has distance 2.0)
        self.assertFalse(np.all(out[:63] == 0.0))
        # Last 63 values must be exactly zeros
        np.testing.assert_array_equal(out[63:], np.zeros(63, dtype=np.float32))

    def test_right_hand_only(self) -> None:
        """Verifies that right hand only populates the last 63 values and leaves the first 63 as zeros."""
        frame = FrameFeatures(left_hand=None, right_hand=self.right_hand)
        out = self.normalizer.normalize(frame)
        self.assertEqual(out.shape, (126,))
        
        # First 63 values must be exactly zeros
        np.testing.assert_array_equal(out[:63], np.zeros(63, dtype=np.float32))
        # Last 63 values should not be all zeros
        self.assertFalse(np.all(out[63:] == 0.0))

    def test_both_hands(self) -> None:
        """Verifies that both hands populate both halves of the output array."""
        frame = FrameFeatures(left_hand=self.left_hand, right_hand=self.right_hand)
        out = self.normalizer.normalize(frame)
        self.assertEqual(out.shape, (126,))
        
        self.assertFalse(np.all(out[:63] == 0.0))
        self.assertFalse(np.all(out[63:] == 0.0))


if __name__ == '__main__':
    unittest.main()
