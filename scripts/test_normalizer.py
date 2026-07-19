"""Test script for demonstrating and verifying the LandmarkNormalizer class."""

import os
import sys
import time
import logging
import cv2
import numpy as np

# Add the workspace root to sys.path to resolve backend imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from backend.sign_recognition.camera import Camera, CameraError
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.normalizer import LandmarkNormalizer

# Set up basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the live webcam test loop verifying landmark normalization."""
    logger.info("Initializing LandmarkNormalizer testing script...")

    try:
        # Initialize pipeline components
        detector = MediaPipeDetector()
        extractor = LandmarkExtractor()
        normalizer = LandmarkNormalizer()
        camera = Camera(camera_index=0)
        camera.start()
    except CameraError as e:
        logger.error("Failed to start camera: %s", e)
        return
    except Exception as e:
        logger.error("Initialization error: %s", e)
        return

    logger.info("Starting live feed window. Press 'Q' to exit.")

    prev_time = time.time()
    last_print_time = prev_time
    fps = 0.0

    try:
        while True:
            # Capture frame
            frame = camera.read_frame()
            if frame is None:
                continue

            current_time = time.time()
            # Calculate FPS
            time_diff = current_time - prev_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
            prev_time = current_time

            # Detect Hands
            detection_res = detector.detect(frame)

            # Extract raw landmarks
            raw_frame_features = extractor.extract(detection_res)

            # Normalize landmarks (returns a (126,) array)
            normalized_vector = normalizer.normalize(raw_frame_features)

            # Split into left and right 63-element halves
            left_half = normalized_vector[:63]
            right_half = normalized_vector[63:]

            # Count non-zeros
            left_non_zero = np.count_nonzero(left_half)
            right_non_zero = np.count_nonzero(right_half)

            # Annotate Frame with Hand Landmarks
            annotated_frame = detector.draw(frame)

            # Draw labels near the wrist
            height, width = frame.shape[:2]
            for h_feat in [raw_frame_features.left_hand, raw_frame_features.right_hand]:
                if h_feat is None:
                    continue
                if len(h_feat.landmarks) >= 2:
                    x_norm = h_feat.landmarks[0]
                    y_norm = h_feat.landmarks[1]
                    px = int(x_norm * width)
                    py = int(y_norm * height)
                    text_x = max(10, min(width - 80, px))
                    text_y = max(25, min(height - 10, py - 20))
                    
                    cv2.putText(annotated_frame, h_feat.handedness, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(annotated_frame, h_feat.handedness, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Overlay information on the video frame
            overlay_texts = [
                f"FPS: {fps:.1f}",
                f"Hands: {detection_res['num_hands']}",
                f"Vector Shape: {normalized_vector.shape}",
                f"Left Non-Zero: {left_non_zero}/63",
                f"Right Non-Zero: {right_non_zero}/63"
            ]

            # Draw overlays
            for idx, text in enumerate(overlay_texts):
                y_pos = 30 + (idx * 30)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            # Show image
            cv2.imshow("SilentVoice Landmark Normalizer Test", annotated_frame)

            # Print stats once per second
            if current_time - last_print_time >= 1.0:
                print("=" * 60)
                print(f"FPS: {fps:.1f}")
                print(f"Normalized vector shape: {normalized_vector.shape}")
                print(f"Left half size: {left_half.shape[0]}, non-zeros: {left_non_zero}")
                print(f"Right half size: {right_half.shape[0]}, non-zeros: {right_non_zero}")
                
                left_first_10 = [f"{val:.4f}" for val in left_half[:10]]
                right_first_10 = [f"{val:.4f}" for val in right_half[:10]]
                
                print("Left Hand (first 10 normalized values):")
                print(f"[{', '.join(left_first_10)}...]")
                print("Right Hand (first 10 normalized values):")
                print(f"[{', '.join(right_first_10)}...]")
                print("=" * 60 + "\n")
                last_print_time = current_time

            # Exit cleanly when Q is pressed
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                logger.info("Exit key pressed.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception:
        logger.exception("An error occurred during execution:")
    finally:
        logger.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        if camera.is_open():
            camera.release()
        detector.close()
        logger.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
