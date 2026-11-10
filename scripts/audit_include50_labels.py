import json
import pandas as pd
from pathlib import Path

base = Path("datasets/include50/processed")

class_names = json.loads(
    (base / "class_names.json").read_text(encoding="utf-8")
)

manifest = pd.read_csv(
    "datasets/metadata_cache/include50_manifest_v1.csv"
)

print("CLASS INDEX MAPPING")
print("-" * 50)

for index, name in enumerate(class_names):
    print(f"{index:02d} -> {name}")

print("\nMANIFEST LABEL CONSISTENCY")
print("-" * 50)

print("Unique manifest labels:", manifest["label"].nunique())
print("All metadata matches path:",
      manifest["hf_label_matches_path"].all())

print("\nLABEL COUNTS BY SPLIT")
print(
    manifest.groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)
