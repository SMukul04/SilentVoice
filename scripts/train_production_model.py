"""Production training entry script for training the SilentVoice LSTM gesture classifier."""

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.models.lstm_model import LSTMModelBuilder
from backend.training.production_trainer import ProductionTrainer

# Configurable production training settings
EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 15
RESUME = True

# Default dataset and checkpoint paths
TRAIN_PATH = "datasets/landmarks/train.npz"
VAL_PATH = "datasets/landmarks/validation.npz"
CHECKPOINT_DIR = "models/checkpoints"
HISTORY_PATH = "artifacts/training_history.json"


def main():
    # Setup argparse to allow overriding defaults from command line if desired
    parser = argparse.ArgumentParser(description="SilentVoice Production Training script")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Patience for early stopping")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch (disable resume)")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    resume_enabled = not args.no_resume if args.no_resume else RESUME

    # Print training configuration report
    print("===================================")
    print("SILENTVOICE PRODUCTION TRAINING")
    print("===================================")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Patience: {patience}")
    print(f"Resume Enabled: {resume_enabled}")
    print(f"Train Dataset: {TRAIN_PATH}")
    print(f"Validation Dataset: {VAL_PATH}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    print("===================================\n")

    # Paths converted to Path objects
    train_file = Path(TRAIN_PATH)
    val_file = Path(VAL_PATH)
    ckpt_dir = Path(CHECKPOINT_DIR)
    history_file = Path(HISTORY_PATH)

    # Initialize model builder with production configurations
    # Note: LSTMModelBuilder uses (32, 126) sequence shape and 13 output classes
    builder = LSTMModelBuilder(
        input_shape=(32, 126),
        num_classes=13,
        learning_rate=0.001,
    )

    # Instantiate production trainer
    trainer = ProductionTrainer(
        model_builder=builder,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        checkpoint_dir=ckpt_dir,
        history_output_path=history_file,
        verbose=1,
    )

    try:
        # Load and validate datasets
        print("Loading and validating datasets...")
        trainer.load_datasets(train_file, val_file)
        print("Datasets loaded and validated successfully.")

        # Check for resumable checkpoints
        checkpoint_exists = trainer.find_resume_checkpoint()
        is_resuming = resume_enabled and checkpoint_exists

        if is_resuming:
            state = trainer.load_training_state()
            completed_epochs = state.get("completed_epochs", 0) if state else 0
            print("\n===================================")
            print("Resuming training from checkpoint...")
            print(f"Completed Epochs: {completed_epochs}")
            print("===================================\n")
        else:
            if resume_enabled:
                print("\nResume enabled, but no valid checkpoint files were found.")
            print("\n===================================")
            print("Starting fresh training run...")
            print("===================================\n")

        # Run training
        model, history = trainer.train(resume=is_resuming)

        # Print training final report
        print("\n===================================")
        print("PRODUCTION TRAINING COMPLETE")
        print("===================================")

        state = trainer.load_training_state()
        final_epoch = state.get("completed_epochs", 0) if state else "N/A"

        print(f"Final Epoch Reached: {final_epoch}")
        print(f"Best Model Path: {trainer.best_model_path}")
        print(f"Latest Checkpoint Path: {trainer.latest_model_path}")
        print(f"Training History Path: {trainer.history_output_path}")

        # Extract final metrics if available
        if history:
            if "accuracy" in history:
                print(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}")
            if "val_accuracy" in history:
                print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
                print(f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
        else:
            # If no history was returned (e.g. resumed when already completed), try reading from the saved JSON history file
            if history_file.exists():
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        saved_history = json.load(f)
                    if "accuracy" in saved_history:
                        print(f"Final Training Accuracy (from history): {saved_history['accuracy'][-1]:.4f}")
                    if "val_accuracy" in saved_history:
                        print(f"Final Validation Accuracy (from history): {saved_history['val_accuracy'][-1]:.4f}")
                        print(f"Best Validation Accuracy (from history): {max(saved_history['val_accuracy']):.4f}")
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n===================================")
        print("TRAINING INTERRUPTED (Ctrl+C)")
        print("===================================")
        print("Training was interrupted by user.")
        print("Checkpoints and state files have been safely preserved.")

        # Print current completed epoch details if available
        state = trainer.load_training_state()
        if state:
            print(f"Completed Epochs: {state.get('completed_epochs', 0)}")
        print("You can resume training by running this script again.")
        print("===================================")
        sys.exit(0)

    except Exception as e:
        print(f"\nTraining pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
