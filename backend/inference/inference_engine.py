"""Real-time inference engine for SilentVoice LSTM sign classification."""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
from typing import Any, Deque, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class InferenceEngine:
    """Runs production LSTM inference over a sliding window of landmark vectors.

    Each frame is a 126-d vector: left hand (21 x 3) concatenated with right hand (21 x 3).
    The trained model expects a sequence of shape (batch_size, sequence_length, 126).
    """

    def __init__(
        self,
        model_path: PathLike = "models/checkpoints/best_model.keras",
        metadata_path: PathLike = "datasets/landmarks/metadata.json",
        sequence_length: int = 32,
        feature_dimension: int = 126,
        confidence_threshold: float = 0.0,
    ) -> None:
        """Initializes inference paths, sequence settings, and an empty landmark buffer.

        Args:
            model_path: Path to the trained Keras model.
            metadata_path: Path to landmark dataset metadata JSON.
            sequence_length: Number of frames required before prediction.
            feature_dimension: Size of each landmark feature vector.
            confidence_threshold: Minimum confidence required to emit a class name.
                Predictions below this threshold return predicted_class \"unknown\".
                Values greater than 1.0 are allowed so callers can force unknown labels.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If an argument is empty or out of range.
        """
        for path_name, val in [("model_path", model_path), ("metadata_path", metadata_path)]:
            if not isinstance(val, (str, Path)) or isinstance(val, bool):
                raise TypeError(f"{path_name} must be a string or Path")
            if str(val).strip() == "":
                raise ValueError(f"{path_name} cannot be empty")

        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise TypeError("sequence_length must be an integer")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")

        if not isinstance(feature_dimension, int) or isinstance(feature_dimension, bool):
            raise TypeError("feature_dimension must be an integer")
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be a positive integer")

        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
            raise TypeError("confidence_threshold must be a float")
        if not np.isfinite(confidence_threshold):
            raise ValueError("confidence_threshold must be a finite number")
        if float(confidence_threshold) < 0.0:
            raise ValueError("confidence_threshold cannot be negative")

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.sequence_length = sequence_length
        self.feature_dimension = feature_dimension
        self.confidence_threshold = float(confidence_threshold)

        self.model = None
        self.class_to_index: dict[str, int] | None = None
        self.index_to_class: dict[int, str] | None = None
        self.classes: list[str] | None = None
        self.num_classes = 0

        self._buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        logger.info(
            "Initialized InferenceEngine (sequence_length=%d, feature_dimension=%d, threshold=%.4f)",
            self.sequence_length,
            self.feature_dimension,
            self.confidence_threshold,
        )

    @property
    def sequence_count(self) -> int:
        """Returns the number of landmark frames currently stored."""
        return len(self._buffer)

    def load_model(self) -> None:
        """Loads the trained Keras model once for reuse during prediction.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model cannot be loaded or has incompatible shapes.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model path is not a file: {self.model_path}")

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            raise ValueError(f"Failed to load Keras model: {e}") from e

        input_shape = getattr(self.model, "input_shape", None)
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(
                f"Model input_shape must be (None, sequence_length, feature_dimension), got {input_shape}"
            )

        model_seq_len = int(input_shape[1])
        model_feat_dim = int(input_shape[2])
        if model_seq_len != self.sequence_length:
            raise ValueError(
                f"Model sequence_length ({model_seq_len}) does not match engine ({self.sequence_length})"
            )
        if model_feat_dim != self.feature_dimension:
            raise ValueError(
                f"Model feature_dimension ({model_feat_dim}) does not match engine ({self.feature_dimension})"
            )

        output_shape = getattr(self.model, "output_shape", None)
        if output_shape is None or len(output_shape) != 2:
            raise ValueError(f"Model output_shape must be (None, num_classes), got {output_shape}")

        model_num_classes = int(output_shape[1])
        if model_num_classes <= 1:
            raise ValueError("Loaded model must have more than one output class")

        if self.num_classes and self.num_classes != model_num_classes:
            raise ValueError(
                f"Metadata class count ({self.num_classes}) does not match model output classes ({model_num_classes})"
            )
        if not self.num_classes:
            self.num_classes = model_num_classes

        logger.info("Loaded Keras model from %s", self.model_path)

    def load_metadata(self) -> None:
        """Loads class mappings from the existing landmark metadata JSON format.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If the JSON is invalid or the class mapping is unusable.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata path is not a file: {self.metadata_path}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid metadata: failed to parse JSON: {e}") from e

        if not isinstance(metadata, dict):
            raise ValueError("Invalid metadata structure: expected a JSON object")
        if "index_to_class" not in metadata or "class_to_index" not in metadata:
            raise ValueError("Invalid metadata structure: class mapping not found")

        raw_class_to_index = metadata["class_to_index"]
        raw_index_to_class = metadata["index_to_class"]
        if not isinstance(raw_class_to_index, dict) or not isinstance(raw_index_to_class, dict):
            raise ValueError("Invalid metadata structure: class mapping must be an object")
        if not raw_class_to_index or not raw_index_to_class:
            raise ValueError("Invalid metadata structure: class mapping is empty")

        try:
            class_to_index = {str(name): int(index) for name, index in raw_class_to_index.items()}
            index_to_class = {int(index): str(name) for index, name in raw_index_to_class.items()}
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid metadata class indices: {e}") from e

        if any(index < 0 for index in index_to_class):
            raise ValueError("Invalid metadata class indices: indices must be non-negative")

        expected_indices = set(range(len(index_to_class)))
        actual_indices = set(index_to_class)
        if actual_indices != expected_indices:
            raise ValueError(
                "Invalid metadata class indices: indices must be contiguous and start at 0"
            )

        classes = [index_to_class[i] for i in sorted(index_to_class.keys())]
        if len(classes) <= 1:
            raise ValueError("Metadata must contain more than one class")

        for name, index in class_to_index.items():
            if index not in index_to_class:
                raise ValueError(f"Invalid metadata class mapping: index {index} for '{name}' is missing")
            if index_to_class[index] != name:
                raise ValueError("Invalid metadata class mapping: class_to_index and index_to_class disagree")

        if len(class_to_index) != len(index_to_class):
            raise ValueError("Invalid metadata class mapping: mapping sizes disagree")

        if "num_classes" in metadata:
            try:
                declared_num_classes = int(metadata["num_classes"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid metadata num_classes: {e}") from e
            if declared_num_classes != len(classes):
                raise ValueError("Metadata num_classes does not match class mapping size")

        if "sequence_length" in metadata:
            metadata_seq_len = int(metadata["sequence_length"])
            if metadata_seq_len != self.sequence_length:
                raise ValueError(
                    f"Metadata sequence_length ({metadata_seq_len}) does not match engine ({self.sequence_length})"
                )

        if "feature_dimension" in metadata:
            metadata_feat_dim = int(metadata["feature_dimension"])
            if metadata_feat_dim != self.feature_dimension:
                raise ValueError(
                    f"Metadata feature_dimension ({metadata_feat_dim}) does not match engine ({self.feature_dimension})"
                )

        if self.num_classes and self.num_classes != len(classes):
            raise ValueError(
                f"Model output classes ({self.num_classes}) do not match metadata class count ({len(classes)})"
            )

        self.class_to_index = class_to_index
        self.index_to_class = index_to_class
        self.classes = classes
        self.num_classes = len(classes)
        logger.info("Loaded metadata with %d classes from %s", self.num_classes, self.metadata_path)

    def load(self) -> None:
        """Loads metadata and the trained model, then cross-validates class counts."""
        self.load_metadata()
        self.load_model()

    def validate_landmarks(self, features: Any) -> np.ndarray:
        """Validates a single landmark feature vector.

        Args:
            features: Landmark vector expected as shape (feature_dimension,).

        Returns:
            np.ndarray: A float32 copy of the validated vector.

        Raises:
            TypeError: If features is None or not numeric array-like data.
            ValueError: If the vector has the wrong shape or contains NaN/Inf values.
        """
        if features is None:
            raise TypeError("Landmark features cannot be None")

        if not isinstance(features, np.ndarray):
            try:
                features = np.asarray(features, dtype=np.float32)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Landmark features must be numeric array-like data: {e}") from e
        else:
            if features.dtype.kind not in ("f", "i", "u"):
                raise TypeError(
                    f"Landmark features must be numeric, got dtype {features.dtype}"
                )
            features = features.astype(np.float32, copy=True)

        if features.ndim != 1:
            raise ValueError(
                f"Landmark features must be 1-dimensional with shape ({self.feature_dimension},), "
                f"got shape {features.shape}"
            )
        if features.shape[0] != self.feature_dimension:
            raise ValueError(
                f"Landmark features must contain {self.feature_dimension} values, got {features.shape[0]}"
            )
        if np.any(np.isnan(features)):
            raise ValueError("Landmark features contain NaN values")
        if np.any(np.isinf(features)):
            raise ValueError("Landmark features contain infinite values")

        return features

    def add_landmarks(self, features: Any) -> None:
        """Validates and appends one landmark vector, dropping the oldest frame when full."""
        vector = self.validate_landmarks(features)
        self._buffer.append(vector)
        logger.debug("Added landmark vector; sequence_count=%d", self.sequence_count)

    def reset(self) -> None:
        """Clears the internal landmark sequence buffer."""
        self._buffer.clear()
        logger.debug("InferenceEngine sequence buffer reset")

    def is_ready(self) -> bool:
        """Returns True when the buffer contains exactly sequence_length frames."""
        return self.sequence_count == self.sequence_length

    def predict(self) -> dict[str, Any]:
        """Runs model inference on the current landmark sequence.

        Returns:
            dict: predicted_index, predicted_class, confidence, and probabilities.

        Raises:
            RuntimeError: If the sequence buffer does not yet contain sequence_length frames,
                or if the model/metadata have not been loaded.
            ValueError: If model outputs are invalid or cannot be mapped to metadata classes.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"InferenceEngine is not ready: need {self.sequence_length} frames, "
                f"currently has {self.sequence_count}"
            )
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call load_model() before predict().")
        if self.index_to_class is None or self.classes is None or self.num_classes <= 1:
            raise RuntimeError("Metadata has not been loaded. Call load_metadata() before predict().")

        sequence = np.stack(list(self._buffer), axis=0).astype(np.float32, copy=False)
        if sequence.shape != (self.sequence_length, self.feature_dimension):
            raise ValueError(
                f"Internal sequence has invalid shape {sequence.shape}; "
                f"expected ({self.sequence_length}, {self.feature_dimension})"
            )

        model_input = sequence.reshape(1, self.sequence_length, self.feature_dimension)
        try:
            raw_probabilities = self.model.predict(model_input, verbose=0)
        except Exception as e:
            raise ValueError(f"Model prediction failed: {e}") from e

        probabilities = self._validate_probabilities(raw_probabilities)
        predicted_index = int(np.argmax(probabilities))
        if predicted_index not in self.index_to_class:
            raise ValueError(f"Predicted class index {predicted_index} is not present in metadata")

        confidence = float(probabilities[predicted_index])
        raw_class = self.index_to_class[predicted_index]
        predicted_class = raw_class if confidence >= self.confidence_threshold else "unknown"

        return {
            "predicted_index": predicted_index,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def _validate_probabilities(self, raw_probabilities: Any) -> list[float]:
        """Validates model output and returns a JSON-compatible probability list."""
        if not isinstance(raw_probabilities, np.ndarray):
            try:
                raw_probabilities = np.asarray(raw_probabilities, dtype=np.float32)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Model produced non-numeric probabilities: {e}") from e

        if raw_probabilities.ndim == 2:
            if raw_probabilities.shape[0] != 1:
                raise ValueError(
                    f"Model prediction batch size must be 1, got {raw_probabilities.shape[0]}"
                )
            raw_probabilities = raw_probabilities[0]
        elif raw_probabilities.ndim != 1:
            raise ValueError(
                f"Model output must have shape (1, {self.num_classes}) or ({self.num_classes},), "
                f"got {raw_probabilities.shape}"
            )

        if raw_probabilities.shape[0] != self.num_classes:
            raise ValueError(
                f"Number of probabilities ({raw_probabilities.shape[0]}) does not match "
                f"number of classes ({self.num_classes})"
            )
        if np.any(np.isnan(raw_probabilities)) or np.any(np.isinf(raw_probabilities)):
            raise ValueError("Model produced NaN or infinite probability values")

        probabilities = [float(value) for value in raw_probabilities.tolist()]
        if any(not np.isfinite(value) for value in probabilities):
            raise ValueError("Model produced non-finite probability values")
        return probabilities
