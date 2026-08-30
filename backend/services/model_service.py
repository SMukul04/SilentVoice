"""Service for loading and sharing the trained SilentVoice inference engine."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Callable

from backend.inference.inference_engine import InferenceEngine


logger = logging.getLogger(__name__)


class ModelService:
    """Lazily load one validated :class:`InferenceEngine` for backend consumers.

    The service owns the engine instance for its own lifetime. Calling
    :meth:`get_inference_engine` repeatedly returns the same loaded instance,
    avoiding an expensive Keras model reload per future API request.
    """

    def __init__(
        self,
        model_path: str | Path = "models/checkpoints/best_model.keras",
        metadata_path: str | Path = "datasets/landmarks/metadata.json",
        confidence_threshold: float = 0.0,
        engine_factory: Callable[..., InferenceEngine] = InferenceEngine,
    ) -> None:
        """Configure the model artifacts and inference-engine factory.

        Args:
            model_path: Path to the trained Keras model.
            metadata_path: Path to the corresponding landmark metadata JSON.
            confidence_threshold: Threshold passed through to InferenceEngine.
            engine_factory: Injectable constructor used primarily for tests.
        """
        if not isinstance(model_path, (str, Path)) or isinstance(model_path, bool) or not str(model_path).strip():
            raise ValueError("model_path must be a non-empty string or Path")
        if not isinstance(metadata_path, (str, Path)) or isinstance(metadata_path, bool) or not str(metadata_path).strip():
            raise ValueError("metadata_path must be a non-empty string or Path")
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.confidence_threshold = confidence_threshold
        self._engine_factory = engine_factory
        self._engine: InferenceEngine | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether this service has successfully loaded an engine."""
        return self._engine is not None

    def load(self) -> InferenceEngine:
        """Load the configured inference engine once and return it.

        Raises:
            FileNotFoundError: If model or metadata artifacts are absent.
            RuntimeError: If engine construction or validation/loading fails.
        """
        if self._engine is not None:
            return self._engine

        with self._load_lock:
            if self._engine is not None:
                return self._engine

            logger.info("Loading SilentVoice inference model from %s", self.model_path)
            try:
                engine = self._engine_factory(
                    model_path=self.model_path,
                    metadata_path=self.metadata_path,
                    confidence_threshold=self.confidence_threshold,
                )
                engine.load()
            except FileNotFoundError:
                logger.exception("SilentVoice model artifacts are missing")
                raise
            except Exception as error:
                logger.exception("Failed to load SilentVoice inference model")
                raise RuntimeError(f"Failed to initialize SilentVoice inference engine: {error}") from error

            self._engine = engine
            logger.info("SilentVoice inference model loaded successfully")
            return self._engine

    def get_inference_engine(self) -> InferenceEngine:
        """Return the single loaded inference engine, loading it on first use."""
        return self.load()
