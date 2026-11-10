import cv2
import pandas as pd
from pathlib import Path

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

MANIFEST = Path("datasets/metadata_cache/include50_manifest_v1.csv")
VIDEO_ROOT = Path("datasets/include50/videos")
OUTPUT = Path("datasets/include50/reports/crop_width_experiment_50.csv")

CROP_RATIOS = {
    "original": 1.00,
    "crop_80": 0.80,
    "crop_70": 0.70,
    "crop_50": 0.50,
}

sampler = FrameSampler(sequence_length=32)

manifest = pd.read_csv(MANIFEST)

selected = (
    manifest.groupby("label", group_keys=False)
    .sample(n=1, random_state=42)
    .sort_values("label")
    .reset_index(drop=True)
)

print("Selected videos:", len(selected))
print("Unique classes:", selected["label"].nunique())

def evaluate(video_path, crop_ratio):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

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

                if crop_ratio < 1.0:
                    h, w = frame.shape[:2]
                    crop_width = int(w * crop_ratio)
                    x1 = (w - crop_width) // 2
                    x2 = x1 + crop_width
                    frame = frame[:, x1:x2]

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

        if valid_frames == 0:
            raise RuntimeError("No valid frames processed")

        return {
            "valid": valid_frames,
            "detected": detected,
            "two_hands": two_hands,
            "hand_total": hand_total,
        }

    finally:
        cap.release()

results = []

for video_index, row in selected.iterrows():
    video_path = VIDEO_ROOT / row["video_path"]

    print(
        f"\n[{video_index + 1}/{len(selected)}] "
        f"{row['label']} | {row['video_path']}"
    )

    if not video_path.is_file():
        print("SKIPPED: File not found")
        continue

    result_row = {
        "label": row["label"],
        "category": row["category"],
        "video_path": row["video_path"],
    }

    try:
        for method, ratio in CROP_RATIOS.items():
            metrics = evaluate(video_path, ratio)

            for key, value in metrics.items():
                result_row[f"{method}_{key}"] = value

            print(
                f"  {method}: "
                f"{metrics['detected']}/{metrics['valid']} "
                f"detected, "
                f"{metrics['two_hands']}/{metrics['valid']} "
                f"two hands"
            )

        results.append(result_row)

    except Exception as exc:
        print("ERROR:", exc)

        # Keep failed videos visible in the report.
        result_row["error"] = str(exc)
        results.append(result_row)

df = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 65)
print("CROP WIDTH EXPERIMENT SUMMARY")
print("=" * 65)

successful = df[
    df["original_valid"].notna()
    & df["crop_80_valid"].notna()
    & df["crop_70_valid"].notna()
    & df["crop_50_valid"].notna()
].copy()

print("Selected videos:", len(selected))
print("Successfully evaluated:", len(successful))
print("Failed/skipped:", len(df) - len(successful))

for method in CROP_RATIOS:
    valid = successful[f"{method}_valid"].sum()

    detected = successful[f"{method}_detected"].sum()
    two_hands = successful[f"{method}_two_hands"].sum()

    print(f"\n{method.upper()}")
    print(f"Detection rate: {detected / valid * 100:.2f}%")
    print(f"Two-hand rate: {two_hands / valid * 100:.2f}%")

print("\nIMPROVEMENT OVER ORIGINAL")

for method in ["crop_80", "crop_70", "crop_50"]:
    original_rate = (
        successful["original_detected"].sum()
        / successful["original_valid"].sum()
    ) * 100

    crop_rate = (
        successful[f"{method}_detected"].sum()
        / successful[f"{method}_valid"].sum()
    ) * 100

    two_original = (
        successful["original_two_hands"].sum()
        / successful["original_valid"].sum()
    ) * 100

    two_crop = (
        successful[f"{method}_two_hands"].sum()
        / successful[f"{method}_valid"].sum()
    ) * 100

    print(
        f"{method}: "
        f"Detection {crop_rate - original_rate:+.2f} pp | "
        f"Two hands {two_crop - two_original:+.2f} pp"
    )

print("\nVIDEOS IMPROVED / UNCHANGED / WORSENED")

for method in ["crop_80", "crop_70", "crop_50"]:
    difference = (
        successful[f"{method}_detected"]
        / successful[f"{method}_valid"]
        -
        successful["original_detected"]
        / successful["original_valid"]
    )

    print(
        f"{method}: "
        f"Improved={(difference > 0).sum()}, "
        f"Unchanged={(difference == 0).sum()}, "
        f"Worsened={(difference < 0).sum()}"
    )

print("\nREPORT SAVED:", OUTPUT)
