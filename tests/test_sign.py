import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.sign_recognition.camera import Camera, CameraError


class TestCamera(unittest.TestCase):
    """Unit tests for the Camera class."""

    def setUp(self) -> None:
        """Sets up a Camera instance before each test."""
        self.camera_index = 0
        self.camera = Camera(camera_index=self.camera_index)

    def tearDown(self) -> None:
        """Cleans up the camera instance after each test."""
        if self.camera.is_open():
            self.camera.release()

    def test_init(self) -> None:
        """Tests that the Camera class initializes attributes correctly."""
        self.assertEqual(self.camera.camera_index, self.camera_index)
        self.assertIsNone(self.camera._cap)
        self.assertFalse(self.camera.is_open())

    @patch('cv2.VideoCapture')
    def test_start_success(self, mock_video_capture: MagicMock) -> None:
        """Tests starting the camera successfully."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        self.camera.start()

        self.assertTrue(self.camera.is_open())
        self.assertEqual(self.camera._cap, mock_cap)
        mock_video_capture.assert_called_once_with(self.camera_index)

    @patch('cv2.VideoCapture')
    def test_start_failure_not_opened(self, mock_video_capture: MagicMock) -> None:
        """Tests start failure when VideoCapture cannot open the device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        with self.assertRaises(CameraError) as context:
            self.camera.start()

        self.assertIn("Could not open camera device", str(context.exception))
        self.assertFalse(self.camera.is_open())
        self.assertIsNone(self.camera._cap)

    @patch('cv2.VideoCapture')
    def test_start_failure_exception(self, mock_video_capture: MagicMock) -> None:
        """Tests start failure when VideoCapture initialization raises an error."""
        mock_video_capture.side_effect = Exception("USB Error")

        with self.assertRaises(CameraError) as context:
            self.camera.start()

        self.assertIn("Failed to initialize VideoCapture", str(context.exception))
        self.assertFalse(self.camera.is_open())
        self.assertIsNone(self.camera._cap)

    @patch('cv2.VideoCapture')
    def test_start_already_open(self, mock_video_capture: MagicMock) -> None:
        """Tests starting a camera that is already open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        self.camera.start()

        with self.assertRaises(CameraError) as context:
            self.camera.start()

        self.assertIn("already open and running", str(context.exception))

    def test_read_frame_when_not_open(self) -> None:
        """Tests reading a frame when the camera has not been started."""
        with self.assertRaises(CameraError) as context:
            self.camera.read_frame()

        self.assertIn("Camera is not open", str(context.exception))

    @patch('cv2.VideoCapture')
    def test_read_frame_success(self, mock_video_capture: MagicMock) -> None:
        """Tests reading a frame successfully when camera is open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_video_capture.return_value = mock_cap

        self.camera.start()
        frame = self.camera.read_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        np.testing.assert_array_equal(frame, dummy_frame)

    @patch('cv2.VideoCapture')
    def test_read_frame_failure(self, mock_video_capture: MagicMock) -> None:
        """Tests reading a frame when VideoCapture read fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        self.camera.start()
        frame = self.camera.read_frame()

        self.assertIsNone(frame)

    @patch('cv2.VideoCapture')
    def test_read_frame_exception(self, mock_video_capture: MagicMock) -> None:
        """Tests reading a frame when read raises an exception."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = Exception("Buffer Overflow")
        mock_video_capture.return_value = mock_cap

        self.camera.start()
        with self.assertRaises(CameraError) as context:
            self.camera.read_frame()

        self.assertIn("Error reading frame from camera", str(context.exception))

    @patch('cv2.VideoCapture')
    def test_release_when_open(self, mock_video_capture: MagicMock) -> None:
        """Tests releasing the camera when it is open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        self.camera.start()
        self.camera.release()

        self.assertFalse(self.camera.is_open())
        self.assertIsNone(self.camera._cap)
        mock_cap.release.assert_called_once()

    def test_release_when_closed(self) -> None:
        """Tests calling release when camera is already closed/not open."""
        # This should not raise an error
        self.camera.release()
        self.assertFalse(self.camera.is_open())
        self.assertIsNone(self.camera._cap)


if __name__ == '__main__':
    unittest.main()
