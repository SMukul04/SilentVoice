"""Tests for backend model-service loading and reuse behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.services.model_service import ModelService


class FakeInferenceEngine:
    """Minimal injectable engine that records construction and loading."""

    created = 0

    def __init__(self, **kwargs: object) -> None:
        type(self).created += 1
        self.kwargs = kwargs
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1


class FailingInferenceEngine(FakeInferenceEngine):
    """Fake engine that simulates an unreadable model artifact."""

    def load(self) -> None:
        raise ValueError("invalid model")


class TestModelService(unittest.TestCase):
    """Verify construction, failure propagation, and singleton engine reuse."""

    def setUp(self) -> None:
        FakeInferenceEngine.created = 0

    def test_initialization_is_lazy(self) -> None:
        service = ModelService(engine_factory=FakeInferenceEngine)
        self.assertFalse(service.is_loaded)
        self.assertEqual(FakeInferenceEngine.created, 0)

    def test_get_engine_loads_once_and_reuses_instance(self) -> None:
        service = ModelService(
            model_path=Path("model.keras"),
            metadata_path=Path("metadata.json"),
            confidence_threshold=0.4,
            engine_factory=FakeInferenceEngine,
        )
        first = service.get_inference_engine()
        second = service.get_inference_engine()
        self.assertIs(first, second)
        self.assertTrue(service.is_loaded)
        self.assertEqual(FakeInferenceEngine.created, 1)
        self.assertEqual(first.load_calls, 1)
        self.assertEqual(first.kwargs["confidence_threshold"], 0.4)

    def test_invalid_service_configuration(self) -> None:
        with self.assertRaises(ValueError):
            ModelService(model_path="", engine_factory=FakeInferenceEngine)
        with self.assertRaises(TypeError):
            ModelService(engine_factory=None)  # type: ignore[arg-type]

    def test_load_failure_has_clear_runtime_error(self) -> None:
        service = ModelService(engine_factory=FailingInferenceEngine)
        with self.assertRaisesRegex(RuntimeError, "Failed to initialize"):
            service.load()
        self.assertFalse(service.is_loaded)


if __name__ == "__main__":
    unittest.main()
