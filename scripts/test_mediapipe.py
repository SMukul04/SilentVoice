"""Standalone testing and debugging tool for MediaPipe hand detection."""

import os
import sys
import time
import logging
import cv2

# Add the workspace root to sys.path to resolve backend imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from backend.sign_recognition.camera import Camera, CameraError
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

# Set up basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the live webcam MediaPipe Hand Detection test loop."""
    logger.info("Initializing testing script...")

    try:
        # Initialize detector and camera (index 0 is default webcam)
        detector = MediaPipeDetector()
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
            # 1. Capture frame
            frame = camera.read_frame()
            if frame is None:
                continue

            current_time = time.time()
            # Calculate FPS
            time_diff = current_time - prev_time
            if time_diff > 0:
                # Use simple moving average/smoothing for FPS to avoid flicker
                current_fps = 1.0 / time_diff
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
            prev_time = current_time

            # 2. Detect Hands
            detection_res = detector.detect(frame)

            # 3. Annotate Frame with Hand Landmarks
            annotated_frame = detector.draw(frame)

            # 4. Prepare information overlays
            num_hands = detection_res["num_hands"]
            handedness = detection_res["handedness"]
            landmarks = detection_res["landmarks"]

            # Overlay information on the video frame
            overlay_texts = [
                f"FPS: {fps:.1f}",
                f"Hands Detected: {num_hands}",
                f"Handedness: {', '.join(handedness) if handedness else 'None'}",
            ]
            for idx, lm_list in enumerate(landmarks):
                overlay_texts.append(f"Hand {idx} ({handedness[idx]}): {len(lm_list)} landmarks")

            # Draw overlays on the frame
            for idx, text in enumerate(overlay_texts):
                y_pos = 30 + (idx * 30)
                # Draw text with shadow for better readability
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            # 5. Show image
            cv2.imshow("SilentVoice MediaPipe Test", annotated_frame)

            # 6. Concise debug print to terminal once per second
            if current_time - last_print_time >= 1.0:
                landmark_counts = [len(lm) for lm in landmarks]
                print(
                    f"[DEBUG] FPS: {fps:.1f} | Hands: {num_hands} | "
                    f"Handedness: {handedness} | Landmarks: {landmark_counts}"
                )
                last_print_time = current_time

            # 7. Check for exit key
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                logger.info("Exit key pressed.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error("An error occurred during execution: %s", e)
    finally:
        # Clean up all resources
        logger.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        if camera.is_open():
            camera.release()
        detector.close()
        logger.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
