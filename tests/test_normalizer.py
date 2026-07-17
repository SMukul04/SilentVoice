import unittest
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.normalizer import LandmarkNormalizer


class TestLandmarkNormalizer(unittest.TestCase):
    """Unit tests for the LandmarkNormalizer class."""

    def setUp(self) -> None:
        """Sets up LandmarkNormalizer and test inputs."""
        self.normalizer = LandmarkNormalizer()

    def test_normalize_translation_and_scaling(self) -> None:
        """Tests that landmarks are correctly translated and scaled."""
        # Create a mock hand: wrist at (1.0, 2.0, 3.0), middle MCP at (1.0, 5.0, 3.0)
        # Expected reference distance before translation is 3.0
        coords = np.zeros((21, 3), dtype=np.float32)
        coords[0] = [1.0, 2.0, 3.0]
        coords[9] = [1.0, 5.0, 3.0]  # Middle MCP

        # fill in other landmarks to avoid all zeros
        for i in range(1, 21):
            if i != 9:
                coords[i] = [float(i), float(i), float(i)]

        hand_features = HandFeatures(
            landmarks=coords.flatten(),
            handedness="Right",
            confidence=0.98
        )

        normalized_features = self.normalizer.normalize(hand_features)

        # Ensure type is HandFeatures and instances are distinct
        self.assertIsInstance(normalized_features, HandFeatures)
        self.assertNotEqual(id(hand_features), id(normalized_features))

        # Check metadata preserved
        self.assertEqual(normalized_features.handedness, "Right")
        self.assertEqual(normalized_features.confidence, 0.98)

        # Check shape
        self.assertEqual(normalized_features.landmarks.shape, (63,))

        # Reshape to check coordinates
        norm_coords = normalized_features.landmarks.reshape(21, 3)

        # Check wrist (landmark 0) is at origin
        np.testing.assert_array_almost_equal(norm_coords[0], [0.0, 0.0, 0.0])

        # Check scale distance:
        # Distance from wrist (now [0,0,0]) to landmark 9 should be exactly 1.0
        dist_9 = np.linalg.norm(norm_coords[9])
        self.assertAlmostEqual(float(dist_9), 1.0, places=5)

    def test_normalize_zero_reference_distance(self) -> None:
        """Tests fallback when the reference distance is near zero."""
        coords = np.zeros((21, 3), dtype=np.float32)
        coords[0] = [1.0, 2.0, 3.0]
        coords[9] = [1.0, 2.0, 3.0]  # Same as wrist

        hand_features = HandFeatures(
            landmarks=coords.flatten(),
            handedness="Left",
            confidence=0.90
        )

        normalized_features = self.normalizer.normalize(hand_features)

        # Wrist should be at origin
        norm_coords = normalized_features.landmarks.reshape(21, 3)
        np.testing.assert_array_almost_equal(norm_coords[0], [0.0, 0.0, 0.0])
        # Since distance is 0, it should not scale, so landmark 9 should also be at [0, 0, 0]
        np.testing.assert_array_almost_equal(norm_coords[9], [0.0, 0.0, 0.0])

    def test_invalid_types_raise_exception(self) -> None:
        """Tests that invalid input type raises TypeError."""
        with self.assertRaises(TypeError):
            self.normalizer.normalize("not_a_hand_features_object")


if __name__ == '__main__':
    unittest.main()
