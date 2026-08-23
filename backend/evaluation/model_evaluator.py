"""Model Evaluator module for verifying LSTM model performance on test datasets."""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    """Evaluates a trained Keras model against test landmark sequence data."""

    def __init__(
        self,
        model_path="models/checkpoints/best_model.keras",
        test_data_path="datasets/landmarks/test.npz",
        metadata_path="datasets/landmarks/metadata.json",
        output_dir="artifacts/evaluation",
    ):
        """Initializes ModelEvaluator paths with validations.

        Args:
            model_path (str/Path): Path to Keras model file.
            test_data_path (str/Path): Path to test .npz dataset file.
            metadata_path (str/Path): Path to metadata JSON file.
            output_dir (str/Path): Output directory for saving reports.
        """
        # Strict validation of paths
        for path_name, val in [
            ("model_path", model_path),
            ("test_data_path", test_data_path),
            ("metadata_path", metadata_path),
            ("output_dir", output_dir),
        ]:
            if not isinstance(val, (str, Path)) or isinstance(val, bool):
                raise TypeError(f"{path_name} must be a string or Path")
            if str(val).strip() == "":
                raise ValueError(f"{path_name} cannot be empty")

        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)

        # Datasets, model, and metadata variables
        self.model = None
        self.X_test = None
        self.y_test = None
        self.class_to_index = None
        self.index_to_class = None
        self.classes = None
        self.num_classes = 0

    def load_model(self):
        """Loads the Keras model.

        Raises:
            FileNotFoundError: If the model file is missing.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            raise ValueError(f"Failed to load Keras model: {e}")

    def load_test_dataset(self):
        """Loads and validates test dataset from .npz file.

        Raises:
            FileNotFoundError: If test dataset is missing.
            ValueError: If dataset format/contents are invalid.
        """
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found at: {self.test_data_path}")

        try:
            with np.load(self.test_data_path) as data:
                if "X" not in data or "y" not in data:
                    raise ValueError("Missing X or y arrays in dataset")
                X_test = data["X"]
                y_test = data["y"]
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Invalid or corrupted .npz dataset: {e}")

        # Validate constraints
        if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
            raise TypeError("Dataset X and y must be numpy.ndarrays")

        if X_test.ndim != 3:
            raise ValueError("Invalid dataset dimensions: X must be 3-dimensional")
        if y_test.ndim != 1:
            raise ValueError("Invalid dataset dimensions: y must be 1-dimensional")

        if len(X_test) != len(y_test):
            raise ValueError("Mismatched X and y lengths")
        if len(X_test) == 0:
            raise ValueError("Empty dataset")

        if np.any(np.isnan(X_test)):
            raise ValueError("Dataset contains NaN values in features (X)")
        if np.any(np.isnan(y_test)):
            raise ValueError("Dataset contains NaN values in labels (y)")

        self.X_test = X_test
        self.y_test = y_test

    def load_metadata(self):
        """Loads and parses the dataset metadata JSON file.

        Raises:
            FileNotFoundError: If metadata file is missing.
            ValueError: If metadata structure is invalid.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid metadata: Failed to parse JSON: {e}")

        if "index_to_class" not in metadata or "class_to_index" not in metadata:
            raise ValueError("Invalid metadata structure: class mapping not found")

        self.class_to_index = metadata["class_to_index"]
        # Convert keys in index_to_class to integers
        try:
            self.index_to_class = {int(k): v for k, v in metadata["index_to_class"].items()}
        except Exception as e:
            raise ValueError(f"Invalid metadata index values: {e}")

        self.classes = [self.index_to_class[i] for i in sorted(self.index_to_class.keys())]
        self.num_classes = len(self.classes)

    def evaluate_loss_and_accuracy(self):
        """Calculates loss and accuracy on the test set.

        Returns:
            tuple: (loss, accuracy) as floats.
        """
        if self.model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model or test dataset not loaded.")
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return float(loss), float(accuracy)

    def generate_predictions(self):
        """Generates class predictions and probabilities.

        Returns:
            tuple: (probabilities, predicted_indices, true_indices)
        """
        if self.model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model or test dataset not loaded.")

        # Predict probabilities
        probabilities = self.model.predict(self.X_test, verbose=0)

        # Validate prediction dimension
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(
                f"Model prediction shape mismatch: expected feature dimension {self.num_classes}, got {probabilities.shape[1]}"
            )

        predicted_indices = np.argmax(probabilities, axis=-1)

        # Validate label ranges
        if np.any(self.y_test < 0) or np.any(self.y_test >= self.num_classes):
            raise ValueError("Invalid class indices in test labels")

        return probabilities, predicted_indices, self.y_test

    def generate_classification_report(self, predicted_indices):
        """Generates detailed precision, recall, f1-score per class.

        Args:
            predicted_indices (np.ndarray): Predictions array.

        Returns:
            dict: Classification report parsed as a dictionary.
        """
        report_dict = classification_report(
            self.y_test,
            predicted_indices,
            labels=list(range(self.num_classes)),
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        return report_dict

    def generate_confusion_matrix(self, predicted_indices):
        """Computes confusion matrix with full labels coverage.

        Args:
            predicted_indices (np.ndarray): Predictions array.

        Returns:
            np.ndarray: Confusion matrix of shape (num_classes, num_classes).
        """
        cm = confusion_matrix(
            self.y_test,
            predicted_indices,
            labels=list(range(self.num_classes)),
        )
        return cm

    def calculate_per_class_accuracy(self, predicted_indices):
        """Calculates accurate per-class accuracies, handling zero total samples safely.

        Args:
            predicted_indices (np.ndarray): Predictions array.

        Returns:
            dict: Mapping from class name to accuracy float.
        """
        per_class_acc = {}
        for idx, class_name in enumerate(self.classes):
            true_mask = self.y_test == idx
            total_samples = np.sum(true_mask)
            if total_samples > 0:
                correct_predictions = np.sum((predicted_indices == idx) & true_mask)
                accuracy = float(correct_predictions / total_samples)
            else:
                accuracy = 0.0
            per_class_acc[class_name] = accuracy
        return per_class_acc

    def save_results(self, test_loss, test_accuracy, per_class_accuracy, report, conf_matrix):
        """Saves evaluation JSON and NPY files.

        Raises:
            OSError: If output directory cannot be created.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Output directory creation issues: {e}")

        # Save evaluation_results.json
        results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "total_test_samples": int(len(self.y_test)),
            "model_path": str(self.model_path),
            "test_data_path": str(self.test_data_path),
            "number_of_classes": int(self.num_classes),
            "per_class_accuracy": per_class_accuracy,
        }
        with open(self.output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        # Save classification_report.json
        with open(self.output_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # Save confusion_matrix.npy
        np.save(self.output_dir / "confusion_matrix.npy", conf_matrix)

    def evaluate(self):
        """Runs the entire evaluation orchestration pipeline.

        Returns:
            dict: Structured dictionary containing the evaluation results summary.
        """
        self.load_model()
        self.load_test_dataset()
        self.load_metadata()

        loss, acc = self.evaluate_loss_and_accuracy()
        _, preds, _ = self.generate_predictions()

        report = self.generate_classification_report(preds)
        cm = self.generate_confusion_matrix(preds)
        per_class_acc = self.calculate_per_class_accuracy(preds)

        self.save_results(loss, acc, per_class_acc, report, cm)

        return {
            "test_loss": loss,
            "test_accuracy": acc,
            "total_test_samples": int(len(self.y_test)),
            "per_class_accuracy": per_class_acc,
        }
