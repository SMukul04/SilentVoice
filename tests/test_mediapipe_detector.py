import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.sign_recognition.mediapipe_detector import MediaPipeDetector


class TestMediaPipeDetector(unittest.TestCase):
    """Unit tests for the MediaPipeDetector class."""

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_initialization(self, mock_hands_class: MagicMock) -> None:
        """Tests that MediaPipe Hands is initialized exactly once with correct parameters."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        detector = MediaPipeDetector()

        # Check that Hands class was instantiated once
        mock_hands_class.assert_called_once()
        self.assertEqual(detector.hands, mock_hands_instance)
        self.assertIsNone(detector._latest_raw_results)

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_detect_empty_frame(self, mock_hands_class: MagicMock) -> None:
        """Tests detect returns empty structure when frame is None."""
        detector = MediaPipeDetector()
        res = detector.detect(None)

        self.assertFalse(res["success"])
        self.assertEqual(res["num_hands"], 0)
        self.assertEqual(res["handedness"], [])
        self.assertEqual(res["landmarks"], [])

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_detect_no_hands(self, mock_hands_class: MagicMock) -> None:
        """Tests detection when no hands are present in the frame."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        # Mock hands.process return value with no landmarks detected
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        mock_results.multi_handedness = None
        mock_hands_instance.process.return_value = mock_results

        detector = MediaPipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        self.assertFalse(res["success"])
        self.assertEqual(res["num_hands"], 0)
        self.assertEqual(res["handedness"], [])
        self.assertEqual(res["landmarks"], [])
        self.assertEqual(detector._latest_raw_results, mock_results)

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_detect_two_hands(self, mock_hands_class: MagicMock) -> None:
        """Tests detection when two hands are present in the frame."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        # 1. Mock Handedness
        mock_handedness_1 = MagicMock()
        mock_handedness_1.classification = [MagicMock(label="Right")]
        mock_handedness_2 = MagicMock()
        mock_handedness_2.classification = [MagicMock(label="Left")]

        # 2. Mock Landmarks
        mock_landmarks_1 = MagicMock()
        mock_landmarks_1.landmark = [MagicMock(x=0.1 * i, y=0.2 * i, z=0.3 * i) for i in range(21)]
        mock_landmarks_2 = MagicMock()
        mock_landmarks_2.landmark = [MagicMock(x=0.5 + 0.01 * i, y=0.6 + 0.01 * i, z=0.7 + 0.01 * i) for i in range(21)]

        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [mock_landmarks_1, mock_landmarks_2]
        mock_results.multi_handedness = [mock_handedness_1, mock_handedness_2]
        mock_hands_instance.process.return_value = mock_results

        detector = MediaPipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with SWAP_HANDEDNESS = True (default)
        res = detector.detect(frame)
        self.assertTrue(res["success"])
        self.assertEqual(res["num_hands"], 2)
        self.assertEqual(res["handedness"], ["Left", "Right"])  # Right -> Left, Left -> Right

        # Test with SWAP_HANDEDNESS = False
        with patch("backend.sign_recognition.mediapipe_detector.SWAP_HANDEDNESS", False):
            res_no_swap = detector.detect(frame)
            self.assertEqual(res_no_swap["handedness"], ["Right", "Left"])

        
        # Verify first hand landmarks
        self.assertEqual(len(res["landmarks"]), 2)
        self.assertEqual(len(res["landmarks"][0]), 21)
        self.assertAlmostEqual(res["landmarks"][0][1][0], 0.1)
        self.assertAlmostEqual(res["landmarks"][0][1][1], 0.2)
        self.assertAlmostEqual(res["landmarks"][0][1][2], 0.3)

        # Verify second hand landmarks
        self.assertEqual(len(res["landmarks"][1]), 21)
        self.assertAlmostEqual(res["landmarks"][1][0][0], 0.5)

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    @patch("backend.sign_recognition.mediapipe_detector.mp_drawing.draw_landmarks")
    def test_draw_with_landmarks(self, mock_draw_landmarks: MagicMock, mock_hands_class: MagicMock) -> None:
        """Tests that drawing utilities are called when landmarks exist."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        detector = MediaPipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Re-use stored results
        mock_results = MagicMock()
        mock_landmarks = MagicMock()
        mock_results.multi_hand_landmarks = [mock_landmarks]
        detector._latest_raw_results = mock_results

        annotated = detector.draw(frame)

        mock_draw_landmarks.assert_called_once()
        self.assertEqual(annotated.shape, frame.shape)

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_close_releases_resources(self, mock_hands_class: MagicMock) -> None:
        """Tests that close calls hands.close() to release system resources."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        detector = MediaPipeDetector()
        detector.close()

        mock_hands_instance.close.assert_called_once()

    @patch("backend.sign_recognition.mediapipe_detector.mp_hands.Hands")
    def test_context_manager(self, mock_hands_class: MagicMock) -> None:
        """Tests context manager protocol __enter__ and __exit__."""
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance

        with MediaPipeDetector() as detector:
            self.assertEqual(detector.hands, mock_hands_instance)

        mock_hands_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
