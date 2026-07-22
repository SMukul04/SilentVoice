from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset.metadata import MetadataManager

metadata_path = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--ai4bharat--INCLUDE"
    / "snapshots"
    / "9530511747c03569465c6c217102ec3bbcce2fc1"
    / "data"
)

metadata = MetadataManager(metadata_path)

print("Train:", len(metadata.get_train_split()))
print("Validation:", len(metadata.get_validation_split()))
print("Test:", len(metadata.get_test_split()))
print("Classes:", metadata.get_num_classes())
print(metadata.get_class_names()[:10])