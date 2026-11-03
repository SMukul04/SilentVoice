import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("datasets/include50/processed/checkpoints")
rows = []

for split_dir in sorted(ROOT.iterdir()):
    if not split_dir.is_dir():
        continue

    for path in split_dir.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            X = data["X"]

        left = X[:, :63]
        right = X[:, 63:]

        left_missing = np.all(left == 0, axis=1)
        right_missing = np.all(right == 0, axis=1)

        rows.append({
            "split": split_dir.name,
            "video": str(X.shape),
            "left_missing_frames": int(left_missing.sum()),
            "right_missing_frames": int(right_missing.sum()),
            "both_missing_frames": int(
                np.sum(left_missing & right_missing)
            ),
            "left_zero_ratio": float(np.mean(left == 0)),
            "right_zero_ratio": float(np.mean(right == 0)),
            "both_hands_missing": bool(
                np.all(left_missing & right_missing)
            )
        })

df = pd.DataFrame(rows)

print("TOTAL VIDEOS:", len(df))

print("\nHAND DETECTION SUMMARY")
for split, group in df.groupby("split"):
    total_frames = len(group) * 32

    left_missing = group["left_missing_frames"].sum()
    right_missing = group["right_missing_frames"].sum()
    both_missing = group["both_missing_frames"].sum()

    print(f"\n{split.upper()}")
    print(f"Left missing:  {left_missing}/{total_frames} "
          f"({left_missing/total_frames*100:.2f}%)")
    print(f"Right missing: {right_missing}/{total_frames} "
          f"({right_missing/total_frames*100:.2f}%)")
    print(f"Both missing:  {both_missing}/{total_frames} "
          f"({both_missing/total_frames*100:.2f}%)")
    print(f"Left zero ratio:  {group['left_zero_ratio'].mean()*100:.2f}%")
    print(f"Right zero ratio: {group['right_zero_ratio'].mean()*100:.2f}%")

print("\nVIDEOS WITH BOTH HANDS MISSING")
print(
    df[df["both_hands_missing"]].to_string(index=False)
    if df["both_hands_missing"].any()
    else "None"
)
