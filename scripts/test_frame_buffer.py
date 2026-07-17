"""Test script demonstrating and validating the sliding-window FrameBuffer."""

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
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.normalizer import LandmarkNormalizer
from backend.sign_recognition.frame_buffer import FrameBuffer

# Set up basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the live webcam test loop verifying landmark sequence buffering."""
    logger.info("Initializing FrameBuffer testing script...")

    try:
        # Initialize pipeline components
        detector = MediaPipeDetector()
        extractor = LandmarkExtractor()
        normalizer = LandmarkNormalizer()
        frame_buffer = FrameBuffer(buffer_size=30)
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
    total_added_frames = 0
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

            # Extract and normalize
            raw_features_list = extractor.extract(detection_res)
            normalized_features_list = [normalizer.normalize(h) for h in raw_features_list]

            # Accumulate the primary hand (first hand detected) in the FrameBuffer
            if normalized_features_list:
                primary_hand = normalized_features_list[0]
                frame_buffer.add(primary_hand)
                total_added_frames += 1

            # Annotate Frame with Hand Landmarks
            annotated_frame = detector.draw(frame)

            # Draw labels near the wrist
            height, width = frame.shape[:2]
            for h_feat in raw_features_list:
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
                f"Buffer Size: {len(frame_buffer)}/30",
                f"Buffer Full: {frame_buffer.is_full()}"
            ]
            for idx, text in enumerate(overlay_texts):
                y_pos = 30 + (idx * 30)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            # Show image
            cv2.imshow("SilentVoice FrameBuffer Test", annotated_frame)

            # Print stats once per second
            if current_time - last_print_time >= 1.0:
                seq = frame_buffer.get_sequence()
                print("=" * 60)
                print(f"Number of detected hands : {len(normalized_features_list)}")
                print(f"Total frames added       : {total_added_frames}")
                print(f"Current buffer size      : {len(frame_buffer)}")
                print(f"Whether buffer is full   : {frame_buffer.is_full()}")
                print(f"Sequence shape           : {seq.shape}")
                if total_added_frames > 30:
                    print("[INFO] Sliding window active: oldest frames are being auto-removed.")
                print("=" * 60 + "\n")
                last_print_time = current_time

            # Exit cleanly when Q is pressed
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                logger.info("Exit key pressed.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error("An error occurred during execution: %s", e)
    finally:
        logger.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        if camera.is_open():
            camera.release()
        detector.close()
        logger.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
