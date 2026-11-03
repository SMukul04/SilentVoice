import numpy as np
import json
from pathlib import Path
from collections import Counter

BASE = Path("datasets/include50/processed")

with open(BASE / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))

for split in ["train", "val", "test"]:
    with np.load(BASE / f"{split}.npz") as data:
        X = data["X"]
        y = data["y"]

    print(f"\n--- {split.upper()} ---")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("NaN values:", np.isnan(X).sum())
    print("Infinite values:", np.isinf(X).sum())
    print("Unique labels:", len(np.unique(y)))
    print("Label range:", y.min(), "to", y.max())
    print("Class distribution:", dict(sorted(Counter(y.tolist()).items())))

    assert X.shape[1:] == (32, 126)
    assert len(X) == len(y)
    assert np.isfinite(X).all()
    assert y.min() >= 0 and y.max() < len(class_names)

print("\nDATA QUALITY VALIDATION PASSED")
