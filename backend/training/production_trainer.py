"""Production Trainer module for SilentVoice.

Includes support for checkpointing and resuming model training.
"""

import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf


class TrainingStateCallback(tf.keras.callbacks.Callback):
    """Custom callback to update and save the training state JSON at the end of each epoch."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        completed_epochs = epoch + 1
        self.trainer.save_training_state(completed_epochs)


class ProductionTrainer:
    """ProductionTrainer coordinates data loading, validations, checkpointing, and resume logic."""

    def __init__(
        self,
        model_builder,
        batch_size=32,
        epochs=50,
        patience=5,
        checkpoint_dir="models/checkpoints",
        history_output_path="artifacts/training_history.json",
        verbose=1,
    ):
        """Initializes the ProductionTrainer parameters with strict validation.

        Args:
            model_builder: Builder instance that provides a Keras model.
            batch_size (int): Batch size. Defaults to 32.
            epochs (int): Target training epochs. Defaults to 50.
            patience (int): Early stopping patience. Defaults to 5.
            checkpoint_dir (str/Path): Checkpoint directory. Defaults to "models/checkpoints".
            history_output_path (str/Path): Training history JSON file. Defaults to "artifacts/training_history.json".
            verbose (int): Verbosity setting. Defaults to 1.
        """
        # Validate model_builder
        if model_builder is None:
            raise TypeError("model_builder cannot be None")
        if not hasattr(model_builder, "build_model") or not hasattr(model_builder, "get_model"):
            raise TypeError("model_builder must have build_model and get_model methods")

        # Validate batch_size
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size <= 0")

        # Validate epochs
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise TypeError("epochs must be an integer")
        if epochs <= 0:
            raise ValueError("epochs <= 0")

        # Validate patience
        if not isinstance(patience, int) or isinstance(patience, bool):
            raise TypeError("patience must be an integer")
        if patience < 0:
            raise ValueError("patience < 0")

        # Validate checkpoint_dir
        if not isinstance(checkpoint_dir, (str, Path)) or isinstance(checkpoint_dir, bool):
            raise TypeError("checkpoint_dir must be a string or Path")
        if str(checkpoint_dir).strip() == "":
            raise ValueError("checkpoint_dir cannot be empty")

        # Validate history_output_path
        if not isinstance(history_output_path, (str, Path)) or isinstance(history_output_path, bool):
            raise TypeError("history_output_path must be a string or Path")
        if str(history_output_path).strip() == "":
            raise ValueError("history_output_path cannot be empty")

        # Validate verbose
        if not isinstance(verbose, (int, bool)):
            raise TypeError("verbose must be an integer or boolean")

        self.model_builder = model_builder
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_output_path = Path(history_output_path)
        self.verbose = int(verbose)

        # File paths derived from checkpoint_dir
        self.best_model_path = self.checkpoint_dir / "best_model.keras"
        self.latest_model_path = self.checkpoint_dir / "latest_model.keras"
        self.state_path = self.checkpoint_dir / "training_state.json"

        # Datasets and state variables
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.model = None

    def validate_datasets(self):
        """Validates shapes, dims, and NaN constraints for the loaded datasets."""
        for name, X, y in [("train", self.X_train, self.y_train), ("validation", self.X_val, self.y_val)]:
            if X is None or y is None:
                raise ValueError(f"Loaded {name} dataset arrays are missing or None")

            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError(f"{name} features (X) and labels (y) must be numpy.ndarrays")

            if X.ndim != 3:
                raise ValueError(f"{name} X must be 3-dimensional (samples, sequence_length, features)")
            if y.ndim != 1:
                raise ValueError(f"{name} y must be 1-dimensional (samples,)")

            if len(X) != len(y):
                raise ValueError(f"{name} sample count mismatch: X has {len(X)}, y has {len(y)}")
            if len(X) == 0:
                raise ValueError(f"{name} dataset arrays cannot be empty")

            if np.any(np.isnan(X)):
                raise ValueError(f"{name} X contains NaN values")
            if np.any(np.isnan(y)):
                raise ValueError(f"{name} y contains NaN values")

    def load_datasets(self, train_path, val_path):
        """Loads train and validation datasets from .npz files.

        Args:
            train_path (str/Path): Training .npz path.
            val_path (str/Path): Validation .npz path.
        """
        train_path = Path(train_path)
        val_path = Path(val_path)

        if not train_path.exists():
            raise FileNotFoundError(f"Training NPZ file not found at: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation NPZ file not found at: {val_path}")

        with np.load(train_path) as data:
            if "X" not in data or "y" not in data:
                raise ValueError("Training NPZ file must contain 'X' and 'y' arrays")
            self.X_train = data["X"]
            self.y_train = data["y"]

        with np.load(val_path) as data:
            if "X" not in data or "y" not in data:
                raise ValueError("Validation NPZ file must contain 'X' and 'y' arrays")
            self.X_val = data["X"]
            self.y_val = data["y"]

        # Validate loaded datasets
        self.validate_datasets()

    def build_model(self):
        """Builds and compiles a fresh model using the model builder."""
        self.model = self.model_builder.get_model()
        return self.model

    def find_resume_checkpoint(self):
        """Checks if a resumable state exists.

        Returns:
            bool: True if latest_model.keras and training_state.json exist and are valid.
        """
        return self.latest_model_path.exists() and self.state_path.exists()

    def load_training_state(self):
        """Loads completed training progress from JSON.

        Returns:
            dict/None: Dictionary of the saved training state or None.
        """
        if not self.state_path.exists():
            return None
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_training_state(self, completed_epochs):
        """Saves current training progress to JSON.

        Args:
            completed_epochs (int): Total completed epochs.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "completed_epochs": int(completed_epochs),
            "total_epochs": int(self.epochs),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    def setup_callbacks(self, initial_epoch):
        """Prepares early stopping, model saving, and state tracking callbacks.

        Args:
            initial_epoch (int): Epoch index at which training starts/resumed.

        Returns:
            list: List of Keras callbacks.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=self.verbose,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.best_model_path),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=self.verbose,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.latest_model_path),
                save_best_only=False,
                verbose=self.verbose,
            ),
            TrainingStateCallback(self),
        ]
        return callbacks

    def train(self, resume=False):
        """Fits the model, handling resume orchestration if requested.

        Args:
            resume (bool): Resume flag.

        Returns:
            tuple: (tf.keras.Model, dict) containing the model and history dictionary.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Datasets not loaded. Call load_datasets() before training.")

        initial_epoch = 0

        # Handle resume scenario
        if resume and self.find_resume_checkpoint():
            state = self.load_training_state()
            if state is not None:
                completed = state.get("completed_epochs", 0)
                if completed >= self.epochs:
                    if self.verbose:
                        print(f"Training already completed up to target epochs ({self.epochs}).")
                    if self.best_model_path.exists():
                        self.model = tf.keras.models.load_model(str(self.best_model_path))
                    else:
                        self.model = tf.keras.models.load_model(str(self.latest_model_path))
                    # Return empty history
                    return self.model, {}
                
                initial_epoch = completed
                if self.verbose:
                    print(f"Resuming training from epoch {initial_epoch}...")
                
                # Load latest checkpoint. Keras saves optimizer state alongside weights.
                self.model = tf.keras.models.load_model(str(self.latest_model_path))
        
        # If not resuming or no checkpoint was found, build a fresh model
        if self.model is None:
            if self.verbose:
                print("Starting training from scratch...")
            self.build_model()

        callbacks = self.setup_callbacks(initial_epoch)

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_epoch=initial_epoch,
            verbose=self.verbose,
            callbacks=callbacks,
        )

        self.save_history(history)
        return self.model, history.history

    def save_history(self, history):
        """Saves Keras history dict to JSON, converting numpy values to native floats.

        Args:
            history: Keras history object.
        """
        if history is None or not hasattr(history, "history"):
            return

        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)

        history_dict = {}
        for metric, values in history.history.items():
            history_dict[metric] = [float(v) for v in values]

        with open(self.history_output_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=4)
