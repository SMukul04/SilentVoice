import cv2
import numpy as np
from pathlib import Path

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

VIDEOS = [
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_4999.MOV"),
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_5000.MOV"),
    Path("datasets/include50/videos/Places/28. Store or Shop/MVI_3387.MOV"),
]

sampler = FrameSampler(sequence_length=32)

def prepare_crop(frame):
    height, width = frame.shape[:2]

    x1 = width // 4
    x2 = 3 * width // 4

    crop = frame[:, x1:x2]

    return cv2.resize(
        crop,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

def evaluate(detector, video_path, mode):
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    indices = [
        int(p.name)
        for p in sampler.sample(
            [Path(str(i)) for i in range(total)]
        )
    ]

    detections = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()

        if not ok:
            detections.append(-1)
            continue

        if mode == "crop":
            frame = prepare_crop(frame)

        timestamp = (
            int(frame_index * 1000 / fps)
            if fps > 0 else frame_index
        )

        detector._timestamp_ms = max(
            detector._timestamp_ms,
            timestamp - 1
        )

        result = detector.detect(frame)
        detections.append(result["num_hands"])

    cap.release()
    return detections

for video in VIDEOS:
    print("\n" + "=" * 65)
    print("VIDEO:", video.name)

    with MediaPipeDetector() as original_detector:
        original = evaluate(
            original_detector, video, "original"
        )

    with MediaPipeDetector() as crop_detector:
        cropped = evaluate(
            crop_detector, video, "crop"
        )

    original_detected = sum(x > 0 for x in original)
    crop_detected = sum(x > 0 for x in cropped)

    original_both = sum(x == 2 for x in original)
    crop_both = sum(x == 2 for x in cropped)

    print(f"Original: {original_detected}/32 frames with hands")
    print(f"Cropped:  {crop_detected}/32 frames with hands")

    print(f"Original: {original_both}/32 frames with two hands")
    print(f"Cropped:  {crop_both}/32 frames with two hands")

    print("\nFRAME-BY-FRAME COMPARISON")
    for i, (a, b) in enumerate(zip(original, cropped)):
        print(f"Frame {i:02d}: Original={a}, Cropped={b}")

print("\nEXPERIMENT COMPLETE")
