"""Model Trainer module for orchestrating training of the LSTM model."""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf


class ModelTrainer:
    """Trainer class for loading data, building/compiling model, training, and saving artifacts."""

    def __init__(
        self,
        model_builder,
        batch_size=32,
        epochs=50,
        patience=5,
        model_output_path="models/silentvoice_lstm.keras",
        history_output_path="artifacts/training_history.json",
        verbose=1,
    ):
        """Initializes and validates ModelTrainer configuration parameters.

        Args:
            model_builder: Builder instance that provides a Keras model.
            batch_size (int): Training batch size. Defaults to 32.
            epochs (int): Number of epochs. Defaults to 50.
            patience (int): EarlyStopping patience. Defaults to 5.
            model_output_path (str/Path): Output path for the model. Defaults to "models/silentvoice_lstm.keras".
            history_output_path (str/Path): Output path for the training history. Defaults to "artifacts/training_history.json".
            verbose (int): Verbosity mode. Defaults to 1.
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

        # Validate model_output_path
        if not isinstance(model_output_path, (str, Path)) or isinstance(model_output_path, bool):
            raise TypeError("model_output_path must be a string or Path")
        if str(model_output_path).strip() == "":
            raise ValueError("model_output_path cannot be empty")

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
        self.model_output_path = Path(model_output_path)
        self.history_output_path = Path(history_output_path)
        self.verbose = int(verbose)

        # Internals to hold datasets, model, and history
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.history = None

    def validate_data(self, X, y):
        """Validates features (X) and labels (y) arrays.

        Args:
            X (np.ndarray): Samples array (3D).
            y (np.ndarray): Labels array (1D).
        """
        # Validate types
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy.ndarray")

        # Validate dimensions
        if X.ndim != 3:
            raise ValueError("X must be a 3-dimensional array (samples, sequence_length, features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array (samples,)")

        # Validate counts
        if len(X) != len(y):
            raise ValueError(f"Sample count mismatch: X has {len(X)} samples, y has {len(y)} samples")
        if len(X) == 0:
            raise ValueError("Data arrays cannot be empty")

        # Validate NaNs
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")

    def load_data(self, train_path, val_path):
        """Loads training and validation datasets from .npz files.

        Args:
            train_path (str/Path): Path to training .npz file.
            val_path (str/Path): Path to validation .npz file.
        """
        train_path = Path(train_path)
        val_path = Path(val_path)

        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset file not found at: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset file not found at: {val_path}")

        # Use context managers to properly load and close npz files
        with np.load(train_path) as data:
            if "X" not in data or "y" not in data:
                raise ValueError("Training NPZ file must contain 'X' and 'y' arrays")
            X_train = data["X"]
            y_train = data["y"]

        with np.load(val_path) as data:
            if "X" not in data or "y" not in data:
                raise ValueError("Validation NPZ file must contain 'X' and 'y' arrays")
            X_val = data["X"]
            y_val = data["y"]

        # Validate loaded arrays
        self.validate_data(X_train, y_train)
        self.validate_data(X_val, y_val)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def build_model(self):
        """Builds and compiles the LSTM model via the model builder."""
        self.model = self.model_builder.get_model()
        return self.model

    def train(self):
        """Trains the model with EarlyStopping on validation loss."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() before train().")
        if self.model is None:
            raise ValueError("Model not built. Call build_model() before train().")

        # Set up early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=self.verbose,
        )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[early_stopping],
        )
        return self.history

    def save_model(self):
        """Saves the trained model to file."""
        if self.model is None:
            raise ValueError("No model exists to save.")

        # Create output directories if necessary
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_output_path))

    def save_history(self):
        """Saves training history to JSON."""
        if self.history is None or not hasattr(self.history, "history"):
            raise ValueError("No training history exists to save.")

        # Create output directories if necessary
        self.history_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 values from numpy to native Python floats for serialization
        history_dict = {}
        for metric, values in self.history.history.items():
            history_dict[metric] = [float(v) for v in values]

        with open(self.history_output_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=4)

    def run(self, train_path, val_path):
        """Runs the complete training pipeline.

        Args:
            train_path (str/Path): Training .npz path.
            val_path (str/Path): Validation .npz path.

        Returns:
            tuple: (tf.keras.Model, dict) containing the trained model and history dict.
        """
        self.load_data(train_path, val_path)
        self.build_model()
        self.train()
        self.save_model()
        self.save_history()
        return self.model, self.history.history
