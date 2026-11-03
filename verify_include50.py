import numpy as np
from pathlib import Path

p = Path("datasets/include50/processed")

for split in ["train", "val", "test"]:
    with np.load(p / f"{split}.npz") as data:
        print(
            split,
            "X:", data["X"].shape,
            "y:", data["y"].shape,
            "dtype:", data["X"].dtype
        )
