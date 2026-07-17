import unittest
import numpy as np

from backend.sign_recognition.landmark_extractor import LandmarkExtractor


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
            "landmarks": [hand_lms]
        }

        vectors = self.extractor.extract(res)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].shape, (63,))
        self.assertEqual(vectors[0].dtype, np.float32)

        # Verify values
        expected_vector = []
        for lm in hand_lms:
            expected_vector.extend(lm)
        np.testing.assert_array_almost_equal(vectors[0], np.array(expected_vector, dtype=np.float32))

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
            "landmarks": [hand0, hand1, hand2, hand3, hand4]
        }

        vectors = self.extractor.extract(res)
        # Should only successfully extract hand0
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].shape, (63,))


if __name__ == '__main__':
    unittest.main()
