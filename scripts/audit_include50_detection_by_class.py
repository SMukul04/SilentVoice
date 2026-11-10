import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path("datasets/include50/processed/checkpoints")
class_names = json.loads(
    Path("datasets/include50/processed/class_names.json")
    .read_text(encoding="utf-8")
)

rows = []

for split_dir in ROOT.iterdir():
    if not split_dir.is_dir():
        continue

    for path in split_dir.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = int(data["y"])
            video = str(data["video_path"])

        left_missing = np.all(X[:, :63] == 0, axis=1)
        right_missing = np.all(X[:, 63:] == 0, axis=1)

        rows.append({
            "split": split_dir.name,
            "label": y,
            "class_name": class_names[y],
            "video": video,
            "both_missing_pct": np.mean(
                left_missing & right_missing
            ) * 100,
            "left_missing_pct": np.mean(left_missing) * 100,
            "right_missing_pct": np.mean(right_missing) * 100
        })

df = pd.DataFrame(rows)

summary = (
    df.groupby("class_name")
    .agg(
        videos=("video", "count"),
        both_missing_avg=("both_missing_pct", "mean"),
        left_missing_avg=("left_missing_pct", "mean"),
        right_missing_avg=("right_missing_pct", "mean")
    )
    .sort_values("both_missing_avg", ascending=False)
)

print("TOP 15 CLASSES WITH HIGHEST BOTH-HANDS-MISSING RATE")
print(summary.head(15).round(2).to_string())

print("\nTOP 10 INDIVIDUAL VIDEOS WITH HIGHEST BOTH-HANDS-MISSING RATE")
print(
    df.sort_values("both_missing_pct", ascending=False)
    [["split", "class_name", "video", "both_missing_pct"]]
    .head(10).round(2).to_string(index=False)
)
