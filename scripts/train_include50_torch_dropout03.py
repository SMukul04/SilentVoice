from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "datasets" / "include50" / "processed"
OUTPUT_ROOT = ROOT / "models" / "include50_torch_dropout03"
MODEL_PATH = OUTPUT_ROOT / "best_model.pth"
HISTORY_PATH = ROOT / "artifacts" / "include50_torch_dropout03" / "training_history.json"
REPORT_PATH = ROOT / "datasets" / "include50" / "reports_dropout03" / "torch_test_evaluation.json"
CONFUSION_PATH = (
    ROOT / "datasets" / "include50" / "reports_dropout03" / "torch_confusion_matrix.csv"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)

    if X.ndim != 3 or X.shape[1:] != (32, 126):
        raise ValueError(f"Invalid feature shape: {X.shape}")

    if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError(f"Invalid label shape: {y.shape}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features found in {path}")

    if np.any(y < 0) or np.any(y >= 50):
        raise ValueError(f"Invalid class labels in {path}")

    return X, y


class SignLSTM(nn.Module):
    """PyTorch equivalent of the existing two-layer Keras LSTM."""

    def __init__(
        self,
        input_features=126,
        hidden_units=64,
        dense_units=64,
        num_classes=50,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_units,
            hidden_size=hidden_units,
            batch_first=True,
        )

        self.dropout2 = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        x = self.dropout2(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Return logits. CrossEntropyLoss applies log-softmax internally.
        return self.output(x)


def make_loader(X, y, batch_size, shuffle, device):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Safe default for Windows.
        pin_memory=(device.type == "cuda"),
    )


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(X)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def evaluate_predictions(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)

            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())

    return np.asarray(targets), np.asarray(predictions)


def calculate_metrics(y_true, y_pred, num_classes=50):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    np.add.at(confusion, (y_true, y_pred), 1)

    per_class = []

    for i in range(num_classes):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0

        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        per_class.append(
            {
                "class_index": i,
                "test_support": support,
                "correct": tp,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": recall if support else None,
            }
        )

    accuracy = float(np.mean(y_true == y_pred))
    macro_precision = float(np.mean([x["precision"] for x in per_class]))
    macro_recall = float(np.mean([x["recall"] for x in per_class]))
    macro_f1 = float(np.mean([x["f1_score"] for x in per_class]))

    return confusion, {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train INCLUDE-50 LSTM using PyTorch CUDA."
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")

    if args.patience < 0 or args.learning_rate <= 0:
        parser.error("Invalid patience or learning rate.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training stopped to prevent CPU fallback."
        )

    device = torch.device("cuda:0")
    set_seed(args.seed)

    print("=" * 60)
    print("INCLUDE-50 | PyTorch CUDA Training")
    print("=" * 60)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2),
        "GB",
    )

    class_path = DATA_ROOT / "class_names.json"

    if not class_path.is_file():
        raise FileNotFoundError(f"Missing class names: {class_path}")

    class_names = json.loads(class_path.read_text(encoding="utf-8"))

    if not isinstance(class_names, list) or len(class_names) != 50:
        raise ValueError("Expected exactly 50 class names.")

    if len(set(class_names)) != 50:
        raise ValueError("Duplicate class names found.")

    X_train, y_train = load_split(DATA_ROOT / "train.npz")
    X_val, y_val = load_split(DATA_ROOT / "val.npz")
    X_test, y_test = load_split(DATA_ROOT / "test.npz")

    print("\nDataset:")
    print("Train:", X_train.shape, "Classes:", len(np.unique(y_train)))
    print("Val:  ", X_val.shape, "Classes:", len(np.unique(y_val)))
    print("Test: ", X_test.shape, "Classes:", len(np.unique(y_test)))

    # Preserve the official split; do not modify or rebalance the test set.
    if not np.array_equal(np.unique(y_train), np.arange(50)):
        raise ValueError("Training split must contain all 50 classes.")

    if not np.array_equal(np.unique(y_val), np.arange(50)):
        raise ValueError("Validation split must contain all 50 classes.")

    if MODEL_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Model already exists: {MODEL_PATH}. "
            "Use --overwrite only if you intend to replace it."
        )

    if HISTORY_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"History already exists: {HISTORY_PATH}. " "Use --overwrite to replace it."
        )

    train_loader = make_loader(X_train, y_train, args.batch_size, True, device)
    val_loader = make_loader(X_val, y_val, args.batch_size, False, device)
    test_loader = make_loader(X_test, y_test, args.batch_size, False, device)

    model = SignLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    print("\nModel parameters:", sum(p.numel() for p in model.parameters()))
    print("Batch size:", args.batch_size)
    print("Maximum epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)
    print("Early stopping patience:", args.patience)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save atomically so an interrupted write doesn't corrupt the model.
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "class_names": class_names,
                "architecture": {
                    "input_features": 126,
                    "hidden_units": 64,
                    "dense_units": 64,
                    "num_classes": 50,
                    "dropout": 0.5,
                },
                "seed": args.seed,
            }

            temp_path = MODEL_PATH.with_name(MODEL_PATH.stem + ".tmp.pth")
            torch.save(checkpoint, temp_path)
            temp_path.replace(MODEL_PATH)

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy * 100:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy * 100:.2f}%"
            f"{' | BEST' if improved else ''}"
        )

        # Persist history after every epoch.
        HISTORY_PATH.write_text(
            json.dumps(
                {
                    **history,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if not MODEL_PATH.exists():
        raise RuntimeError("Training finished without a saved best model.")

    # Restore the best validation-loss checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nBest epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["best_val_loss"])

    # The official test set is evaluated only after model selection.
    test_targets, test_predictions = evaluate_predictions(model, test_loader, device)

    confusion, metrics = calculate_metrics(test_targets, test_predictions)

    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, None, device)

    np.savetxt(
        CONFUSION_PATH,
        confusion,
        delimiter=",",
        fmt="%d",
    )

    report = {
        "dataset": "INCLUDE-50",
        "framework": "PyTorch",
        "device": torch.cuda.get_device_name(device),
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "test_samples": int(len(y_test)),
        "test_classes_present": int(len(np.unique(y_test))),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {**entry, "class_name": class_names[entry["class_index"]]}
            for entry in metrics["per_class"]
        ],
        "notes": [
            "Best checkpoint selected using validation loss.",
            "Official test split evaluated after model selection.",
            "Absent test classes have zero contribution to the 50-class macro metrics; review per-class support.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    print(f"Test loss:       {test_loss:.4f}")
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")

    print("\nSaved files:")
    print("Model:", MODEL_PATH.relative_to(ROOT))
    print("History:", HISTORY_PATH.relative_to(ROOT))
    print("Report:", REPORT_PATH.relative_to(ROOT))
    print("Confusion matrix:", CONFUSION_PATH.relative_to(ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
