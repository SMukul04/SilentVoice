from pathlib import Path
import pandas as pd

# Path to the downloaded metadata
DATASET_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--ai4bharat--INCLUDE"
    / "snapshots"
    / "9530511747c03569465c6c217102ec3bbcce2fc1"
    / "data"
)

train = pd.read_parquet(DATASET_PATH / "train-00000-of-00001.parquet")
val = pd.read_parquet(DATASET_PATH / "val-00000-of-00001.parquet")
test = pd.read_parquet(DATASET_PATH / "test-00000-of-00001.parquet")

print("=" * 60)
print("TRAIN")
print("=" * 60)
print(train.head())

print("\nColumns:")
print(train.columns.tolist())

print("\nShape:")
print(train.shape)

print("\nInfo:")
print(train.info())

print("\nFirst Sample:")
print(train.iloc[0])

print("\n" + "=" * 60)
print("INCLUDE-50")
print("=" * 60)

include50 = train[train["include_50"]]

print(f"Total INCLUDE-50 videos : {len(include50)}")
print(f"Unique labels           : {include50['label'].nunique()}")

print("\nVideos per label:")
print(include50["label"].value_counts().sort_index())