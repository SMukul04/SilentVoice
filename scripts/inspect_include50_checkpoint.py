import numpy as np
from pathlib import Path

root = Path("datasets/include50/processed/checkpoints")
files = sorted(root.rglob("*.npz"))

if not files:
    raise FileNotFoundError("No checkpoint files found.")

path = files[0]

print("Checkpoint:", path)
print("\nStored arrays:")

with np.load(path, allow_pickle=False) as data:
    for key in data.files:
        arr = data[key]
        print(f"\n{key}:")
        print("  Shape:", arr.shape)
        print("  Dtype:", arr.dtype)

        if np.issubdtype(arr.dtype, np.number):
            print("  Min:", np.nanmin(arr))
            print("  Max:", np.nanmax(arr))
            print("  NaN:", np.isnan(arr).sum())
            print("  Inf:", np.isinf(arr).sum())
