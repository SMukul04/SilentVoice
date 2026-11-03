"""PyTorch inference engine for the INCLUDE-50 classifier."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from backend.inference.include50_torch_model import SignLSTM


class TorchInferenceEngine:
    """Load and run the trained INCLUDE-50 PyTorch model."""

    def __init__(
        self,
        model_path="models/include50_torch_dropout03/best_model.pth",
        metadata_path="datasets/include50/processed/class_names.json",
        confidence_threshold=0.0,
        sequence_length=32,
        feature_dimension=126,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not 0.0 <= float(confidence_threshold):
            raise ValueError("confidence_threshold cannot be negative")

        self.confidence_threshold = float(confidence_threshold)
        self.sequence_length = int(sequence_length)
        self.feature_dimension = int(feature_dimension)

        if self.sequence_length <= 0 or self.feature_dimension <= 0:
            raise ValueError("Sequence length and feature dimension must be positive")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.index_to_class = {}
        self.num_classes = 0

        self._buffer = deque(maxlen=self.sequence_length)

    @property
    def sequence_count(self):
        return len(self._buffer)

    def load(self):
        """Load class mapping and trained model checkpoint."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Class mapping not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8-sig") as file:
            class_names = json.load(file)

        if (
            not isinstance(class_names, list)
            or not class_names
            or not all(isinstance(name, str) and name.strip() for name in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError("Class mapping must be a non-empty list of unique class names")

        self.class_names = class_names
        self.index_to_class = {
            index: name for index, name in enumerate(class_names)
        }
        self.num_classes = len(class_names)

        self.model = SignLSTM(
            input_features=self.feature_dimension,
            hidden_units=64,
            dense_units=64,
            num_classes=self.num_classes,
            dropout=0.3,
        ).to(self.device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model_state_dict")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.reset()
        return self

    def add_landmarks(self, landmarks):
        """Add one normalized frame of landmark features."""
        if self.model is None:
            raise RuntimeError("Load the model before adding landmarks")

        features = np.asarray(landmarks, dtype=np.float32)

        if features.shape != (self.feature_dimension,):
            raise ValueError(
                f"Expected landmark shape ({self.feature_dimension},), "
                f"received {features.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError("Landmarks contain NaN or infinite values")

        self._buffer.append(features.copy())
        return self.is_ready()

    def is_ready(self):
        """Return whether a complete sequence has been collected."""
        return len(self._buffer) == self.sequence_length

    def reset(self):
        """Clear the current frame sequence."""
        self._buffer.clear()

    def predict(self):
        """Predict the sign represented by the current frame sequence."""
        if self.model is None:
            raise RuntimeError("Load the model before prediction")

        if not self.is_ready():
            raise RuntimeError(
                f"Sequence is incomplete: {self.sequence_count}/{self.sequence_length}"
            )

        sequence = np.stack(self._buffer, axis=0)
        model_input = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(model_input)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        if probabilities.shape != (self.num_classes,):
            raise ValueError("Model returned an unexpected probability shape")

        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned invalid probabilities")

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        predicted_class = (
            self.index_to_class[predicted_index]
            if confidence >= self.confidence_threshold
            else "unknown"
        )

        return {
            "predicted_index": predicted_index,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.astype(float).tolist(),
        }
