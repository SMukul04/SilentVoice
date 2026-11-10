import cv2
import numpy as np
from pathlib import Path

from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.dataset.frame_sampler import FrameSampler

VIDEOS = [
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_4999.MOV"),
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_5000.MOV"),
    Path("datasets/include50/videos/Places/28. Store or Shop/MVI_3387.MOV"),
]

OUTPUT = Path("datasets/include50/reports/visual_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

sampler = FrameSampler(sequence_length=32)

CELL_W, CELL_H = 320, 220
COLS, ROWS = 8, 4

with MediaPipeDetector() as detector:

    for video_path in VIDEOS:
        print(f"\nProcessing: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("ERROR: Cannot open video")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            print("ERROR: Invalid frame count")
            cap.release()
            continue

        sampled = sampler.sample(
            [Path(str(i)) for i in range(total_frames)]
        )

        indices = [int(p.name) for p in sampled]

        sheet = np.zeros(
            (ROWS * CELL_H, COLS * CELL_W, 3),
            dtype=np.uint8
        )

        for i, frame_index in enumerate(indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            source_ms = (
                int(frame_index * 1000 / fps)
                if fps > 0 else frame_index
            )

            timestamp_ms = max(
                source_ms,
                detector._timestamp_ms + 1
            )

            detector._timestamp_ms = timestamp_ms - 1

            detection = detector.detect(frame)
            annotated = detector.draw(frame)

            count = detection["num_hands"]

            annotated = cv2.resize(
                annotated,
                (CELL_W, CELL_H - 30)
            )

            x = (i % COLS) * CELL_W
            y = (i // COLS) * CELL_H

            sheet[
                y:y + CELL_H - 30,
                x:x + CELL_W
            ] = annotated

            label = (
                f"Frame {frame_index} | "
                f"{source_ms / 1000:.2f}s | "
                f"Hands: {count}"
            )

            cv2.putText(
                sheet,
                label,
                (x + 5, y + CELL_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cap.release()

        output_file = OUTPUT / f"{video_path.stem}_contact_sheet.jpg"

        cv2.imwrite(
            str(output_file),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        print("Saved:", output_file)

print("\nVISUAL AUDIT COMPLETE")
