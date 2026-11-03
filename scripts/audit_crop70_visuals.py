import cv2
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("datasets/include50/reports/crop_width_experiment_50.csv")
video_root = Path("datasets/include50/videos")
output = Path("datasets/include50/reports/crop70_visual_audit.png")

df = pd.read_csv(csv_path)

regressed = df[
    (df["crop_70_detected"] < df["original_detected"]) |
    (df["crop_70_two_hands"] < df["original_two_hands"])
].copy()

print("Videos selected:", len(regressed))

tile_w, tile_h = 480, 300
frames_per_video = 4
rows = len(regressed)
cols = frames_per_video

sheet = np.full(
    (rows * tile_h, cols * tile_w, 3),
    245, dtype=np.uint8
)

for row_index, (_, row) in enumerate(regressed.iterrows()):
    video_path = video_root / row["video_path"]
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path)
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), frames_per_video, dtype=int)

    for col_index, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok:
            continue

        h, w = frame.shape[:2]
        crop_width = int(w * 0.70)
        x1 = (w - crop_width) // 2
        x2 = x1 + crop_width

        # Mark the crop boundaries on the original frame.
        cv2.rectangle(frame, (x1, 0), (x2, h), (0, 255, 0), 5)

        # Resize while preserving aspect ratio.
        scale = min(tile_w / w, (tile_h - 55) / h)
        resized = cv2.resize(
            frame,
            (int(w * scale), int(h * scale))
        )

        canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)

        x_offset = (tile_w - resized.shape[1]) // 2
        y_offset = 45 + (tile_h - 45 - resized.shape[0]) // 2

        canvas[
            y_offset:y_offset + resized.shape[0],
            x_offset:x_offset + resized.shape[1]
        ] = resized

        title = f"Frame {frame_index}"
        cv2.putText(
            canvas, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2
        )

        x = col_index * tile_w
        y = row_index * tile_h
        sheet[y:y + tile_h, x:x + tile_w] = canvas

    cap.release()

    label = (
        f"{row['label']} | {row['video_path'].split('/')[-1]} | "
        f"Original: {row['original_detected']}/32, "
        f"Crop70: {row['crop_70_detected']}/32"
    )

    cv2.putText(
        sheet, label,
        (10, row_index * tile_h + tile_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 2
    )

output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), sheet)

print("Saved:", output)
