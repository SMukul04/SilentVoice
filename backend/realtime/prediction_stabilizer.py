"""Prediction smoothing independent of camera and model implementations."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np


class PredictionStabilizer:
    """Average recent class probabilities and gate results by consistency."""

    REQUIRED_KEYS = {"predicted_index", "predicted_class", "confidence", "probabilities"}

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.0, min_consistent_predictions: int = 1) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be an integer greater than or equal to 1")
        if not isinstance(min_consistent_predictions, int) or isinstance(min_consistent_predictions, bool) or min_consistent_predictions < 1:
            raise ValueError("min_consistent_predictions must be an integer greater than or equal to 1")
        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
            raise TypeError("confidence_threshold must be numeric")
        if not np.isfinite(confidence_threshold) or not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be finite and between 0.0 and 1.0")

        self.window_size = window_size
        self.confidence_threshold = float(confidence_threshold)
        self.min_consistent_predictions = min_consistent_predictions
        self._history: Deque[dict[str, Any]] = deque(maxlen=window_size)
        self._probability_count: int | None = None

    @property
    def history_size(self) -> int:
        """Return the number of predictions currently retained."""
        return len(self._history)

    def add_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Validate, retain, and stabilize one InferenceEngine prediction."""
        validated = self._validate_prediction(prediction)
        probability_count = len(validated["probabilities"])
        if self._probability_count is not None and probability_count != self._probability_count:
            raise ValueError(
                f"Incompatible probability vector length: expected {self._probability_count}, got {probability_count}"
            )
        self._probability_count = probability_count
        self._history.append(validated)
        return self._stabilized_result()

    def reset(self) -> None:
        """Clear retained predictions and their established vector shape."""
        self._history.clear()
        self._probability_count = None

    def is_stable(self) -> bool:
        """Return whether the current averaged winner meets both stability gates."""
        if not self._history:
            return False
        result = self._stabilized_result()
        return result["predicted_class"] != "unknown"

    def _validate_prediction(self, prediction: Any) -> dict[str, Any]:
        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a dictionary")
        missing = self.REQUIRED_KEYS - set(prediction)
        if missing:
            raise ValueError(f"prediction is missing required keys: {', '.join(sorted(missing))}")

        index = prediction["predicted_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("predicted_index must be an integer")
        class_name = prediction["predicted_class"]
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError("predicted_class must be a non-empty string")
        confidence = prediction["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
            raise ValueError("confidence must be a finite numeric value")

        try:
            probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"probabilities must be a numeric sequence: {error}") from error
        if probabilities.ndim != 1:
            raise ValueError("probabilities must be a one-dimensional numeric sequence")
        if probabilities.size == 0:
            raise ValueError("probabilities cannot be empty")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any(probabilities < 0):
            raise ValueError("probabilities cannot contain negative values")
        if index < 0 or index >= probabilities.size:
            raise ValueError("predicted_index must be within the probabilities range")

        return {"predicted_index": int(index), "predicted_class": class_name.strip(), "confidence": float(confidence), "probabilities": [float(value) for value in probabilities]}

    def _stabilized_result(self) -> dict[str, Any]:
        matrix = np.asarray([entry["probabilities"] for entry in self._history], dtype=np.float64)
        averaged = matrix.mean(axis=0)
        index = int(np.argmax(averaged))
        confidence = float(averaged[index])
        matching = [entry for entry in self._history if entry["predicted_index"] == index]
        class_name = matching[-1]["predicted_class"] if matching else "unknown"
        if len(matching) < self.min_consistent_predictions or confidence < self.confidence_threshold:
            class_name = "unknown"
        return {"predicted_index": index, "predicted_class": str(class_name), "confidence": confidence, "probabilities": [float(value) for value in averaged]}
