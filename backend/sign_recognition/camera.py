import logging
from typing import Optional
import cv2
import numpy as np

# Configure logger for the camera module
logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Custom exception raised for errors related to camera operations."""
    pass


class Camera:
    """Manages the webcam capture and retrieval of frames using OpenCV.

    This class provides a modular interface to interact with a webcam. It handles
    opening the camera device, reading raw frames, checking if the camera is open,
    and safely releasing the camera resources.

    Attributes:
        camera_index (int): Index of the video capture device to open.
        _cap (Optional[cv2.VideoCapture]): The VideoCapture object from OpenCV.
    """

    def __init__(self, camera_index: int = 0) -> None:
        """Initializes the Camera instance.

        Args:
            camera_index (int): The index of the camera. Defaults to 0.
        """
        self.camera_index: int = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        logger.info(
            "Initialized Camera instance with camera_index %d.",
            self.camera_index
        )

    def start(self) -> None:
        """Opens the camera device.

        Raises:
            CameraError: If the camera is already open or if OpenCV fails
                to open the specified camera device.
        """
        if self._cap is not None and self._cap.isOpened():
            logger.warning("Camera is already open and running.")
            raise CameraError("Camera is already open and running.")

        logger.info(
            "Starting camera device with camera_index %d...",
            self.camera_index
        )
        try:
            self._cap = cv2.VideoCapture(self.camera_index)
        except Exception as e:
            logger.error("Failed to initialize cv2.VideoCapture: %s", e)
            self._cap = None
            raise CameraError(f"Failed to initialize VideoCapture: {e}") from e

        if not self._cap.isOpened():
            logger.error(
                "Could not open camera device with camera_index %d.",
                self.camera_index
            )
            self._cap = None
            raise CameraError(
                f"Could not open camera device at index {self.camera_index}."
            )

        logger.info(
            "Successfully started camera device with camera_index %d.",
            self.camera_index
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a frame from the camera.

        Returns:
            Optional[np.ndarray]: The captured frame as a NumPy array (BGR format),
                or None if the frame could not be read.

        Raises:
            CameraError: If the camera is not currently open/started.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.error(
                "Attempted to read frame from a camera that is not open."
            )
            raise CameraError(
                "Camera is not open. Call start() before reading frames."
            )

        try:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(
                    "Failed to retrieve frame from camera index %d.",
                    self.camera_index
                )
                return None
            return frame
        except Exception as e:
            logger.error("Error reading frame from camera: %s", e)
            raise CameraError(f"Error reading frame from camera: {e}") from e

    def is_open(self) -> bool:
        """Checks if the camera device is open.

        Returns:
            bool: True if the camera device is initialized and open, False otherwise.
        """
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Releases the camera capture device if open."""
        if self._cap is not None:
            logger.info(
                "Releasing camera device with camera_index %d...",
                self.camera_index
            )
            self._cap.release()
            self._cap = None
            logger.info("Camera device released successfully.")
        else:
            logger.debug("Release called but camera is not open.")
