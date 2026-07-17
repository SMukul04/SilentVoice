import unittest
import numpy as np

from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.hand_features import HandFeatures


class TestLandmarkExtractor(unittest.TestCase):
    """Unit tests for the LandmarkExtractor class."""

    def setUp(self) -> None:
        """Sets up a LandmarkExtractor instance before each test."""
        self.extractor = LandmarkExtractor()

    def test_extract_empty_or_failed_results(self) -> None:
        """Tests that extract returns an empty list for failed or empty results."""
        # Case 1: success is False
        res1 = {
            "success": False,
            "num_hands": 0,
            "handedness": [],
            "confidences": [],
            "landmarks": []
        }
        self.assertEqual(self.extractor.extract(res1), [])

        # Case 2: Empty dict
        self.assertEqual(self.extractor.extract({}), [])

        # Case 3: Missing landmarks key
        res3 = {"success": True}
        self.assertEqual(self.extractor.extract(res3), [])

    def test_extract_valid_landmarks(self) -> None:
        """Tests extracting features from valid hand landmarks."""
        # Create a mock detection result with 1 hand having 21 sequential landmark coords
        hand_lms = [(float(i), float(i + 1), float(i + 2)) for i in range(21)]
        res = {
            "success": True,
            "num_hands": 1,
            "handedness": ["Right"],
            "confidences": [0.95],
            "landmarks": [hand_lms]
        }

        features = self.extractor.extract(res)
        self.assertEqual(len(features), 1)
        self.assertIsInstance(features[0], HandFeatures)
        self.assertEqual(features[0].landmarks.shape, (63,))
        self.assertEqual(features[0].landmarks.dtype, np.float32)
        self.assertEqual(features[0].handedness, "Right")
        self.assertEqual(features[0].confidence, 0.95)

        # Verify values
        expected_vector = []
        for lm in hand_lms:
            expected_vector.extend(lm)
        np.testing.assert_array_almost_equal(features[0].landmarks, np.array(expected_vector, dtype=np.float32))

    def test_extract_invalid_landmarks_warning(self) -> None:
        """Tests that malformed landmark lists (not 21 elements) are skipped without crashing."""
        # Hand 0 is correct, Hand 1 has only 20 landmarks, Hand 2 has 22
        hand0 = [(float(i), float(i), float(i)) for i in range(21)]
        hand1 = [(float(i), float(i), float(i)) for i in range(20)]
        hand2 = [(float(i), float(i), float(i)) for i in range(22)]
        hand3 = "invalid_type"
        hand4 = [(float(i), float(i)) for i in range(21)]  # Invalid landmark tuple length (2 instead of 3)

        res = {
            "success": True,
            "num_hands": 5,
            "handedness": ["Right", "Left", "Right", "Left", "Right"],
            "confidences": [0.95, 0.9, 0.8, 0.7, 0.6],
            "landmarks": [hand0, hand1, hand2, hand3, hand4]
        }

        features = self.extractor.extract(res)
        # Should only successfully extract hand0
        self.assertEqual(len(features), 1)
        self.assertIsInstance(features[0], HandFeatures)
        self.assertEqual(features[0].landmarks.shape, (63,))
        self.assertEqual(features[0].handedness, "Right")
        self.assertEqual(features[0].confidence, 0.95)

    def test_hand_features_validation(self) -> None:
        """Tests that HandFeatures validates its initialization parameters."""
        valid_lms = np.zeros(63, dtype=np.float32)

        # Valid initialization should succeed
        hf = HandFeatures(landmarks=valid_lms, handedness="Left", confidence=0.8)
        self.assertEqual(hf.handedness, "Left")
        self.assertEqual(hf.confidence, 0.8)

        # Invalid landmarks type
        with self.assertRaises(TypeError):
            HandFeatures(landmarks=[0.0] * 63, handedness="Left", confidence=0.8)

        # Invalid landmarks shape
        with self.assertRaises(ValueError):
            HandFeatures(landmarks=np.zeros(62, dtype=np.float32), handedness="Left", confidence=0.8)

        # Invalid handedness type
        with self.assertRaises(TypeError):
            HandFeatures(landmarks=valid_lms, handedness=123, confidence=0.8)

        # Invalid handedness value
        with self.assertRaises(ValueError):
            HandFeatures(landmarks=valid_lms, handedness="InvalidHand", confidence=0.8)

        # Invalid confidence type
        with self.assertRaises(TypeError):
            HandFeatures(landmarks=valid_lms, handedness="Left", confidence="high")

        # Invalid confidence value range
        with self.assertRaises(ValueError):
            HandFeatures(landmarks=valid_lms, handedness="Left", confidence=1.2)


if __name__ == '__main__':
    unittest.main()
