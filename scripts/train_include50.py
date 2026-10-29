from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models.lstm_model import LSTMModelBuilder
from backend.training.trainer import ModelTrainer

DATA_ROOT = ROOT / "datasets" / "include50" / "processed"
MODEL_PATH = ROOT / "models" / "include50" / "best_model.keras"
HISTORY_PATH = ROOT / "artifacts" / "include50" / "training_history.json"
REPORT_PATH = ROOT / "datasets" / "include50" / "reports" / "test_evaluation.json"
CLASS_NAMES_PATH = DATA_ROOT / "class_names.json"


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed split not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "X" not in data or "y" not in data:
            raise ValueError(f"{path} must contain arrays named 'X' and 'y'.")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)

    if X.ndim != 3 or X.shape[1:] != (32, 126):
        raise ValueError(f"{path}: expected X shape (N, 32, 126), got {X.shape}")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{path}: incompatible X/y shapes: {X.shape}, {y.shape}")
    if len(y) == 0:
        raise ValueError(f"{path}: no samples.")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains NaN or infinite values.")
    if np.any(y < 0) or np.any(y >= 50):
        raise ValueError(f"{path}: labels must be integers from 0 to 49.")
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a separate 50-class INCLUDE-50 LSTM."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the INCLUDE-50 model/history/report outputs only.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("epochs and batch-size must be positive; patience cannot be negative.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")

    required_outputs = (MODEL_PATH, HISTORY_PATH, REPORT_PATH)
    existing = [p for p in required_outputs if p.exists()]
    if existing and not args.overwrite:
        details = "\n".join(f"  {p.relative_to(ROOT)}" for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing INCLUDE-50 outputs:\n"
            f"{details}\nUse --overwrite only if you intend to replace these INCLUDE-50 files."
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CLASS_NAMES_PATH}. Run preprocessing successfully first."
        )
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) != 50:
        raise ValueError("class_names.json must contain exactly 50 class names.")
    if len(set(class_names)) != 50:
        raise ValueError("class_names.json contains duplicate class names.")

    X_train, y_train = load_split(DATA_ROOT / "train.npz")
    X_val, y_val = load_split(DATA_ROOT / "val.npz")
    X_test, y_test = load_split(DATA_ROOT / "test.npz")

    print("INCLUDE-50 dataset loaded")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, classes={len(np.unique(y_train))}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}, classes={len(np.unique(y_val))}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, classes={len(np.unique(y_test))}")
    print(f"  TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")

    # Keep all outputs separate from the existing 13-class SilentVoice model.
    builder = LSTMModelBuilder(
        input_shape=(32, 126),
        num_classes=50,
        lstm_units=(64, 64),
        dropout_rate=0.5,
        dense_units=64,
        learning_rate=args.learning_rate,
    )
    trainer = ModelTrainer(
        model_builder=builder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        model_output_path=MODEL_PATH,
        history_output_path=HISTORY_PATH,
    )
    trainer.load_data(DATA_ROOT / "train.npz", DATA_ROOT / "val.npz")
    trainer.build_model()
    trainer.train()
    trainer.save_model()
    trainer.save_history()

    # Evaluate the restored best-validation weights on the held-out official test split.
    test_loss, test_accuracy = trainer.model.evaluate(
        X_test, y_test, batch_size=args.batch_size, verbose=1
    )
    probabilities = trainer.model.predict(
        X_test, batch_size=args.batch_size, verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)

    confusion = np.zeros((50, 50), dtype=np.int64)
    for actual, predicted in zip(y_test, predictions):
        confusion[int(actual), int(predicted)] += 1

    per_class = []
    for class_index, class_name in enumerate(class_names):
        support = int(np.sum(y_test == class_index))
        correct = int(confusion[class_index, class_index])
        per_class.append({
            "class_index": class_index,
            "class_name": class_name,
            "test_support": support,
            "correct": correct,
            "accuracy": (correct / support) if support else None,
        })

    report = {
        "dataset": "INCLUDE-50",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "class_names": str(CLASS_NAMES_PATH.relative_to(ROOT)),
        "test_samples": int(len(y_test)),
        "test_classes_present": int(len(np.unique(y_test))),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "notes": [
            "Test metrics are computed on the processed test split supplied to this script.",
            "Classes absent from the official test split have null per-class accuracy.",
            "Review preprocessing_report.json for any videos that failed preprocessing.",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nTraining and evaluation complete.")
    print(f"Best model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Training history: {HISTORY_PATH.relative_to(ROOT)}")
    print(f"Test report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
