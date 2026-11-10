import cv2
import pandas as pd
import numpy as np
from pathlib import Path

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

CSV_INPUT = Path("datasets/include50/reports/crop_width_experiment_50.csv")
VIDEO_ROOT = Path("datasets/include50/videos")
OUTPUT = Path("datasets/include50/reports/crop70_frame_diagnostic.csv")

CROP_RATIO = 0.70
EDGE_MARGIN_RATIO = 0.05

df = pd.read_csv(CSV_INPUT)

regressed = df[
    (df["crop_70_detected"] < df["original_detected"]) |
    (df["crop_70_two_hands"] < df["original_two_hands"])
].copy()

sampler = FrameSampler(sequence_length=32)
records = []

def landmark_summary(landmarks, offset_x, scale_x, frame_width):
    if not landmarks:
        return {
            "center_x": None,
            "min_x": None,
            "max_x": None,
            "near_crop_edge": None,
        }

    xs = np.array([point[0] for point in landmarks], dtype=float)

    # Map normalized coordinates back to original frame coordinates.
    xs = (offset_x + xs * scale_x) / frame_width

    min_x = float(xs.min())
    max_x = float(xs.max())
    center_x = float(xs.mean())

    crop_left = (1.0 - CROP_RATIO) / 2
    crop_right = 1.0 - crop_left
    margin = CROP_RATIO * EDGE_MARGIN_RATIO

    near_edge = (
        min_x <= crop_left + margin
        or max_x >= crop_right - margin
    )

    return {
        "center_x": center_x,
        "min_x": min_x,
        "max_x": max_x,
        "near_crop_edge": near_edge,
    }

print("Videos to inspect:", len(regressed))

for video_number, (_, row) in enumerate(regressed.iterrows(), 1):
    video_path = VIDEO_ROOT / row["video_path"]

    print(
        f"\n[{video_number}/{len(regressed)}] "
        f"{row['label']} | {video_path.name}"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("SKIPPED: Cannot open video")
        continue

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            print("SKIPPED: Invalid frame count")
            continue

        indices = [
            int(p.name)
            for p in sampler.sample(
                [Path(str(i)) for i in range(total)]
            )
        ]

        with MediaPipeDetector() as original_detector, \
             MediaPipeDetector() as crop_detector:

            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()

                if not ok or frame is None:
                    continue

                h, w = frame.shape[:2]

                crop_width = int(w * CROP_RATIO)
                x1 = (w - crop_width) // 2

                original_result = original_detector.detect(frame)

                cropped_frame = frame[:, x1:x1 + crop_width]
                crop_result = crop_detector.detect(cropped_frame)

                original_count = original_result["num_hands"]
                crop_count = crop_result["num_hands"]

                transition = "unchanged"

                if original_count > 0 and crop_count == 0:
                    transition = "lost_after_crop"
                elif original_count == 0 and crop_count > 0:
                    transition = "gained_after_crop"
                elif crop_count > original_count:
                    transition = "more_hands_after_crop"
                elif crop_count < original_count:
                    transition = "fewer_hands_after_crop"

                crop_hands = [
                    landmark_summary(
                        hand,
                        x1,
                        crop_width,
                        w
                    )
                    for hand in crop_result["landmarks"]
                ]

                edge_hands = sum(
                    item["near_crop_edge"] is True
                    for item in crop_hands
                )

                records.append({
                    "label": row["label"],
                    "category": row["category"],
                    "video_path": row["video_path"],
                    "frame_index": frame_index,
                    "original_hands": original_count,
                    "crop70_hands": crop_count,
                    "transition": transition,
                    "crop_hands_near_edge": edge_hands,
                    "crop_hand_centres": ";".join(
                        f"{item['center_x']:.4f}"
                        for item in crop_hands
                    ),
                    "crop_hand_min_x": ";".join(
                        f"{item['min_x']:.4f}"
                        for item in crop_hands
                    ),
                    "crop_hand_max_x": ";".join(
                        f"{item['max_x']:.4f}"
                        for item in crop_hands
                    ),
                    "crop_hand_edge_flags": ";".join(
                        str(item["near_crop_edge"])
                        for item in crop_hands
                    ),
                })

    finally:
        cap.release()

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("FRAME-LEVEL DIAGNOSTIC SUMMARY")
print("=" * 60)

print("Total frames:", len(result))

if not result.empty:
    print("\nDetection transitions:")
    print(result["transition"].value_counts().to_string())

    print("\nCropped detected hands near crop boundaries:")
    print(
        result["crop_hands_near_edge"].sum(),
        "hands"
    )

    print("\nPer-video transition summary:")
    print(
        result.groupby(["label", "video_path", "transition"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

print("\nReport saved:", OUTPUT)
