import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

report_path = ROOT / "datasets/include50/reports/torch_test_evaluation.json"
matrix_path = ROOT / "datasets/include50/reports/torch_confusion_matrix.csv"

report = json.loads(report_path.read_text(encoding="utf-8"))
confusion = np.loadtxt(matrix_path, delimiter=",", dtype=int)

class_names = [item["class_name"] for item in report["per_class"]]

# Collect incorrect predictions.
errors = []

for actual in range(50):
    for predicted in range(50):
        count = int(confusion[actual, predicted])

        if actual != predicted and count > 0:
            errors.append(
                {
                    "actual": class_names[actual],
                    "predicted": class_names[predicted],
                    "count": count,
                }
            )

errors.sort(key=lambda item: item["count"], reverse=True)

print("\nTOP 20 CONFUSIONS")
print("-" * 75)

for item in errors[:20]:
    print(
        f"Actual: {item['actual']:<30} "
        f"Predicted: {item['predicted']:<30} "
        f"Count: {item['count']}"
    )

# Calculate macro metrics only for classes represented in test.
represented = [item for item in report["per_class"] if item["test_support"] > 0]

print("\nMETRICS")
print("-" * 75)
print("Overall accuracy:", report["test_accuracy_percent"])
print("Macro F1 (all 50 classes):", report["macro_f1"])
print(
    "Macro F1 (represented classes):",
    np.mean([item["f1_score"] for item in represented]),
)
print("Represented test classes:", len(represented))
