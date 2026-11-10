import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

MANIFEST = Path("datasets/metadata_cache/include50_manifest_v1.csv")
VIDEO_ROOT = Path("datasets/include50/videos")
OUTPUT = Path("datasets/include50/reports/crop_experiment_50.csv")

sampler = FrameSampler(sequence_length=32)

# Select exactly one reproducible video per class.
manifest = pd.read_csv(MANIFEST)

selected = (
    manifest.groupby("label", group_keys=False)
    .sample(n=1, random_state=42)
    .sort_values("label")
    .reset_index(drop=True)
)

print("Selected videos:", len(selected))
print("Unique classes:", selected["label"].nunique())

def evaluate(video_path, crop):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total <= 0:
            raise RuntimeError("Invalid frame count")

        indices = [
            int(p.name)
            for p in sampler.sample(
                [Path(str(i)) for i in range(total)]
            )
        ]

        detected = 0
        two_hands = 0
        hand_total = 0
        valid_frames = 0

        with MediaPipeDetector() as detector:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()

                if not ok or frame is None:
                    continue

                if crop:
                    h, w = frame.shape[:2]
                    frame = frame[:, w // 4: 3 * w // 4]

                timestamp = (
                    int(index * 1000 / fps)
                    if fps > 0 else index
                )

                detector._timestamp_ms = max(
                    detector._timestamp_ms,
                    timestamp - 1
                )

                result = detector.detect(frame)
                count = result["num_hands"]

                detected += count > 0
                two_hands += count == 2
                hand_total += count
                valid_frames += 1

        return {
            "valid_frames": valid_frames,
            "detected_frames": detected,
            "two_hand_frames": two_hands,
            "total_hands": hand_total,
            "detection_rate": detected / valid_frames * 100,
            "two_hand_rate": two_hands / valid_frames * 100,
            "average_hands": hand_total / valid_frames
        }

    finally:
        cap.release()

results = []

for i, row in selected.iterrows():
    video_path = VIDEO_ROOT / row["video_path"]

    print(
        f"\n[{i + 1}/{len(selected)}] "
        f"{row['label']} | {row['video_path']}"
    )

    if not video_path.is_file():
        print("SKIPPED: Video not found")
        continue

    try:
        original = evaluate(video_path, crop=False)
        cropped = evaluate(video_path, crop=True)

        result = {
            "label": row["label"],
            "category": row["category"],
            "video_path": row["video_path"]
        }

        for key, value in original.items():
            result[f"original_{key}"] = value

        for key, value in cropped.items():
            result[f"crop_{key}"] = value

        result["detection_improvement_pp"] = (
            cropped["detection_rate"] -
            original["detection_rate"]
        )

        result["two_hand_improvement_pp"] = (
            cropped["two_hand_rate"] -
            original["two_hand_rate"]
        )

        results.append(result)

        print(
            f"Original: {original['detected_frames']}/"
            f"{original['valid_frames']} detected | "
            f"Crop: {cropped['detected_frames']}/"
            f"{cropped['valid_frames']} detected"
        )

    except Exception as exc:
        print("ERROR:", exc)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print("Videos evaluated:", len(df))

if not df.empty:
    print(
        "Original detection rate:",
        round(
            df["original_detected_frames"].sum() /
            df["original_valid_frames"].sum() * 100, 2
        ), "%"
    )

    print(
        "Cropped detection rate:",
        round(
            df["crop_detected_frames"].sum() /
            df["crop_valid_frames"].sum() * 100, 2
        ), "%"
    )

    print(
        "Original two-hand rate:",
        round(
            df["original_two_hand_frames"].sum() /
            df["original_valid_frames"].sum() * 100, 2
        ), "%"
    )

    print(
        "Cropped two-hand rate:",
        round(
            df["crop_two_hand_frames"].sum() /
            df["crop_valid_frames"].sum() * 100, 2
        ), "%"
    )

    print(
        "Videos improved:",
        int((df["detection_improvement_pp"] > 0).sum())
    )

    print(
        "Videos unchanged:",
        int((df["detection_improvement_pp"] == 0).sum())
    )

    print(
        "Videos worsened:",
        int((df["detection_improvement_pp"] < 0).sum())
    )

    print("\nBIGGEST REGRESSIONS:")
    print(
        df.sort_values("detection_improvement_pp")
        [["label", "video_path", "detection_improvement_pp"]]
        .head(10).to_string(index=False)
    )

print("\nREPORT SAVED:", OUTPUT)
