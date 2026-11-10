from pathlib import Path
import json
import torch
import torch.nn as nn

from scripts import train_include50_torch_dropout03 as trainer

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda:0")

    if not trainer.MODEL_PATH.is_file():
        raise FileNotFoundError(trainer.MODEL_PATH)

    trainer.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(
        (trainer.DATA_ROOT / "class_names.json").read_text(encoding="utf-8")
    )

    X_test, y_test = trainer.load_split(trainer.DATA_ROOT / "test.npz")
    test_loader = trainer.make_loader(
        X_test, y_test, batch_size=16, shuffle=False, device=device
    )

    model = trainer.SignLSTM().to(device)

    checkpoint = torch.load(
        trainer.MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    targets, predictions = trainer.evaluate_predictions(
        model, test_loader, device
    )

    confusion, metrics = trainer.calculate_metrics(targets, predictions)

    test_loss, test_accuracy = trainer.run_epoch(
        model, test_loader, criterion, None, device
    )

    trainer.CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    np.savetxt(trainer.CONFUSION_PATH, confusion, delimiter=",", fmt="%d")

    report = {
        "dataset": "INCLUDE-50",
        "framework": "PyTorch",
        "device": torch.cuda.get_device_name(device),
        "model": str(trainer.MODEL_PATH.relative_to(trainer.ROOT)),
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "test_samples": len(y_test),
        "test_classes_present": len(np.unique(y_test)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                **entry,
                "class_name": class_names[entry["class_index"]]
            }
            for entry in metrics["per_class"]
        ],
    }

    trainer.REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n========== EVALUATION COMPLETE ==========")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Report:", trainer.REPORT_PATH)
    print("Confusion matrix:", trainer.CONFUSION_PATH)

if __name__ == "__main__":
    main()
