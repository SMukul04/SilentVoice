"""Test script for validating and demonstrating the LandmarkExtractor class."""

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

# Set up basic logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Runs the LandmarkExtractor testing loop using live webcam feed."""
    logger.info("Initializing LandmarkExtractor testing script...")

    try:
        # Initialize detector, extractor, and camera
        detector = MediaPipeDetector()
        extractor = LandmarkExtractor()
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

            # Extract Landmarks using LandmarkExtractor
            frame_features = extractor.extract(detection_res)

            # Annotate Frame with Hand Landmarks
            annotated_frame = detector.draw(frame)

            # Overlay information on the video frame
            hands_count = int(frame_features.has_left_hand()) + int(frame_features.has_right_hand())
            overlay_texts = [
                f"FPS: {fps:.1f}",
                f"Hands: {detection_res['num_hands']}",
                f"Extracted Hands: {hands_count}"
            ]
            if frame_features.left_hand is not None:
                h_feat = frame_features.left_hand
                overlay_texts.append(
                    f"Left Shape: {h_feat.landmarks.shape}"
                )

            if frame_features.right_hand is not None:
                h_feat = frame_features.right_hand
                overlay_texts.append(
                    f"Right Shape: {h_feat.landmarks.shape}"
                )

            # Draw overlays
            for idx, text in enumerate(overlay_texts):
                y_pos = 30 + (idx * 30)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw handedness labels above the wrist of each detected hand
            height, width = frame.shape[:2]

            for h_feat in [frame_features.left_hand, frame_features.right_hand]:
                if h_feat is None:
                    continue

                if len(h_feat.landmarks) >= 2:
                    x_norm = h_feat.landmarks[0]
                    y_norm = h_feat.landmarks[1]

                    px = int(x_norm * width)
                    py = int(y_norm * height)

                    text_x = max(10, min(width - 80, px))
                    text_y = max(25, min(height - 10, py - 20))

                    cv2.putText(
                        annotated_frame,
                        h_feat.handedness,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated_frame,
                        h_feat.handedness,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Show annotated image
            cv2.imshow("SilentVoice Landmark Extractor Test", annotated_frame)

            # Print output once per second
            if current_time - last_print_time >= 1.0:
                print("=" * 60)
                print(f"FPS: {fps:.1f}")
                print(f"Detected Hands : {detection_res['num_hands']}")
                print(f"Left Hand      : {'Present' if frame_features.has_left_hand() else 'None'}")
                print(f"Right Hand     : {'Present' if frame_features.has_right_hand() else 'None'}")
                print(f"Both Hands     : {frame_features.has_both_hands()}")
                print()
                for label, h_feat in [
                    ("Left", frame_features.left_hand),
                    ("Right", frame_features.right_hand),
                ]:
                    if h_feat is None:
                        print(f"{label} Hand : None")
                        print()
                        continue
                    formatted_values = [f"{val:.4f}" for val in h_feat.landmarks[:10]]
                    print(f"{label} Hand")
                    print("-" * 25)
                    print(f"Handedness : {h_feat.handedness}")
                    print(f"Confidence : {h_feat.confidence:.2f}")
                    print(f"Shape      : {h_feat.landmarks.shape}")
                    print("First 10 values:")
                    print(f"[{', '.join(formatted_values)}...]")
                    print()
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
