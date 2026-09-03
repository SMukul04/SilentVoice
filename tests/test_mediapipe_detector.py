import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from backend.sign_recognition.mediapipe_detector import MediaPipeDetector


class TestMediaPipeDetector(unittest.TestCase):
    """Unit tests for the MediaPipeDetector class."""

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_initialization(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests that MediaPipe HandLandmarker is initialized exactly once with correct parameters."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        detector = MediaPipeDetector()

        mock_hand_landmarker.create_from_options.assert_called_once()
        self.assertEqual(detector.hand_landmarker, mock_instance)
        self.assertIsNone(detector._latest_result)

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_detect_empty_frame(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests detect returns empty structure when frame is None."""
        detector = MediaPipeDetector()
        res = detector.detect(None)

        self.assertFalse(res["success"])
        self.assertEqual(res["num_hands"], 0)
        self.assertEqual(res["handedness"], [])
        self.assertEqual(res["landmarks"], [])

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_detect_no_hands(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests detection when no hands are present in the frame."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        # Mock detect_for_video return value with no landmarks detected
        mock_results = MagicMock()
        mock_results.hand_landmarks = []
        mock_results.handedness = []
        mock_instance.detect_for_video.return_value = mock_results

        detector = MediaPipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        self.assertFalse(res["success"])
        self.assertEqual(res["num_hands"], 0)
        self.assertEqual(res["handedness"], [])
        self.assertEqual(res["landmarks"], [])
        self.assertEqual(detector._latest_result, mock_results)

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_detect_two_hands(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests detection when two hands are present in the frame."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        # Mock results
        mock_results = MagicMock()

        # 1. Mock Landmarks
        mock_hand_1 = [MagicMock(x=0.1 * i, y=0.2 * i, z=0.3 * i) for i in range(21)]
        mock_hand_2 = [MagicMock(x=0.5 + 0.01 * i, y=0.6 + 0.01 * i, z=0.7 + 0.01 * i) for i in range(21)]
        mock_results.hand_landmarks = [mock_hand_1, mock_hand_2]

        # 2. Mock Handedness
        mock_cat_1 = MagicMock(category_name="Right", score=0.9)
        mock_cat_2 = MagicMock(category_name="Left", score=0.8)
        mock_results.handedness = [[mock_cat_1], [mock_cat_2]]

        mock_instance.detect_for_video.return_value = mock_results

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

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    @patch("backend.sign_recognition.mediapipe_detector.cv2.circle")
    def test_draw_with_landmarks(self, mock_cv2_circle: MagicMock, mock_hand_landmarker: MagicMock) -> None:
        """Tests that drawing utilities are called when landmarks exist."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        detector = MediaPipeDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Re-use stored results
        mock_results = MagicMock()
        mock_hand_landmarks = [MagicMock(x=0.5, y=0.5, z=0.5) for _ in range(21)]
        mock_results.hand_landmarks = [mock_hand_landmarks]
        detector._latest_result = mock_results

        annotated = detector.draw(frame)

        mock_cv2_circle.assert_called()
        self.assertEqual(annotated.shape, frame.shape)

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_close_releases_resources(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests that close calls close() to release system resources."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        detector = MediaPipeDetector()
        detector.close()

        mock_instance.close.assert_called_once()

    @patch("backend.sign_recognition.mediapipe_detector.vision.HandLandmarker")
    def test_context_manager(self, mock_hand_landmarker: MagicMock) -> None:
        """Tests context manager protocol __enter__ and __exit__."""
        mock_instance = MagicMock()
        mock_hand_landmarker.create_from_options.return_value = mock_instance

        with MediaPipeDetector() as detector:
            self.assertEqual(detector.hand_landmarker, mock_instance)

        mock_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
