import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("datasets/include50/processed/checkpoints")
OUTPUT = Path("datasets/include50/reports/landmark_quality_audit.csv")

results = []

for split_dir in sorted(ROOT.iterdir()):
    if not split_dir.is_dir():
        continue

    for path in split_dir.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as data:
                X = data["X"]
                y = int(data["y"])
                video_path = str(data["video_path"])

            if X.shape != (32, 126):
                raise ValueError(f"Unexpected shape: {X.shape}")

            finite = np.isfinite(X)
            zero_ratio = float(np.mean(X == 0))
            nan_count = int(np.isnan(X).sum())
            inf_count = int(np.isinf(X).sum())

            frame_zero_ratio = np.mean(X == 0, axis=1)
            feature_std = float(np.std(X))
            frame_std = np.std(X, axis=1)

            results.append({
                "split": split_dir.name,
                "checkpoint": path.name,
                "video_path": video_path,
                "label": y,
                "zero_ratio": zero_ratio,
                "mean_frame_zero_ratio": float(np.mean(frame_zero_ratio)),
                "max_frame_zero_ratio": float(np.max(frame_zero_ratio)),
                "feature_std": feature_std,
                "mean_frame_std": float(np.mean(frame_std)),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "finite_ratio": float(np.mean(finite)),
                "status": "OK" if nan_count == 0 and inf_count == 0 else "INVALID"
            })

        except Exception as exc:
            results.append({
                "split": split_dir.name,
                "checkpoint": path.name,
                "video_path": "",
                "label": -1,
                "status": f"ERROR: {exc}"
            })

df = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("TOTAL CHECKPOINTS:", len(df))
print("\nSTATUS COUNTS:")
print(df["status"].value_counts().to_string())

print("\nSPLIT SUMMARY:")
print(
    df.groupby("split").agg(
        videos=("checkpoint", "count"),
        avg_zero_ratio=("zero_ratio", "mean"),
        avg_feature_std=("feature_std", "mean"),
        invalid_values=("nan_count", "sum")
    ).round(4).to_string()
)

print("\nHIGHEST ZERO-RATIO VIDEOS:")
print(
    df.sort_values("zero_ratio", ascending=False)
      [["split", "video_path", "zero_ratio", "feature_std"]]
      .head(15).to_string(index=False)
)

print("\nLOWEST VARIATION VIDEOS:")
print(
    df.sort_values("feature_std")
      [["split", "video_path", "zero_ratio", "feature_std"]]
      .head(15).to_string(index=False)
)

print("\nREPORT SAVED:", OUTPUT)
