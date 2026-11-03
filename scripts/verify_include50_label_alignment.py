import json
import numpy as np
import pandas as pd
from pathlib import Path

base = Path("datasets/include50/processed")

class_names = json.loads(
    (base / "class_names.json").read_text(encoding="utf-8")
)

manifest = pd.read_csv(
    "datasets/metadata_cache/include50_manifest_v1.csv"
)

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

all_passed = True

for split in ["train", "val", "test"]:
    data = np.load(base / f"{split}.npz")
    y = data["y"]

    split_df = manifest[manifest["split"] == split]
    expected = np.zeros(len(class_names), dtype=int)

    for label, count in split_df["label"].value_counts().items():
        expected[class_to_index[label]] = count

    actual = np.bincount(y.astype(int), minlength=len(class_names))

    passed = (
        len(y) == len(split_df)
        and np.array_equal(actual, expected)
        and np.all((y >= 0) & (y < len(class_names)))
    )

    print(f"\n{split.upper()}")
    print("NPZ samples:", len(y))
    print("Manifest samples:", len(split_df))
    print("Label distributions match:", np.array_equal(actual, expected))
    print("Result:", "PASS" if passed else "FAIL")

    if not passed:
        all_passed = False
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                print(f"Mismatch: {i} | {class_names[i]} | NPZ={a} | Manifest={e}")

print("\nFINAL RESULT:", "ALL SPLITS PASSED" if all_passed else "VERIFICATION FAILED")
