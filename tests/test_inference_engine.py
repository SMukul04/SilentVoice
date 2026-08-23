"""Unit and integration tests for the SilentVoice real-time inference engine."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from backend.inference.inference_engine import InferenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.keras"
METADATA_PATH = PROJECT_ROOT / "datasets" / "landmarks" / "metadata.json"
TEST_DATA_PATH = PROJECT_ROOT / "datasets" / "landmarks" / "test.npz"


class TestInferenceEngine(unittest.TestCase):
    """Verifies validation, buffering, and production-model inference."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the local production artifacts once for integration coverage."""
        for required_path in (MODEL_PATH, METADATA_PATH, TEST_DATA_PATH):
            if not required_path.is_file():
                raise unittest.SkipTest(f"Required SilentVoice artifact is missing: {required_path}")

        cls.engine = InferenceEngine(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
        cls.engine.load()
        with np.load(TEST_DATA_PATH) as dataset:
            cls.real_sequence = dataset["X"][0].astype(np.float32, copy=True)
            cls.real_label = int(dataset["y"][0])

    def setUp(self) -> None:
        self.engine.reset()

    def test_component_initialization(self) -> None:
        engine = InferenceEngine()
        self.assertEqual(engine.model_path, Path("models/checkpoints/best_model.keras"))
        self.assertEqual(engine.metadata_path, Path("datasets/landmarks/metadata.json"))
        self.assertEqual(engine.sequence_length, 32)
        self.assertEqual(engine.feature_dimension, 126)
        self.assertEqual(engine.confidence_threshold, 0.0)
        self.assertEqual(engine.sequence_count, 0)

    def test_model_and_metadata_loading(self) -> None:
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.index_to_class)
        self.assertEqual(self.engine.num_classes, 13)
        self.assertEqual(self.engine.model.input_shape, (None, 32, 126))
        self.assertEqual(self.engine.model.output_shape, (None, 13))

    def test_invalid_landmark_input(self) -> None:
        invalid_vectors = (
            None,
            np.zeros(125, dtype=np.float32),
            np.zeros((1, 126), dtype=np.float32),
            ["invalid"] * 126,
            np.full(126, np.nan, dtype=np.float32),
            np.full(126, np.inf, dtype=np.float32),
        )

        for invalid_vector in invalid_vectors:
            with self.subTest(invalid_vector=repr(invalid_vector)[:50]):
                with self.assertRaises((TypeError, ValueError)) as raised:
                    self.engine.add_landmarks(invalid_vector)
                self.assertTrue(str(raised.exception))

    def test_sequence_buffer_discards_oldest_frame(self) -> None:
        for value in range(33):
            self.engine.add_landmarks(np.full(126, value, dtype=np.float32))

        self.assertEqual(self.engine.sequence_count, 32)
        self.assertTrue(self.engine.is_ready())
        self.assertEqual(len(self.engine._buffer), 32)
        np.testing.assert_array_equal(self.engine._buffer[0], np.full(126, 1, dtype=np.float32))
        np.testing.assert_array_equal(self.engine._buffer[-1], np.full(126, 32, dtype=np.float32))

    def test_readiness_behavior(self) -> None:
        vector = np.zeros(126, dtype=np.float32)
        for _ in range(31):
            self.engine.add_landmarks(vector)
        self.assertFalse(self.engine.is_ready())
        self.engine.add_landmarks(vector)
        self.assertTrue(self.engine.is_ready())

    def test_reset_clears_sequence(self) -> None:
        self.engine.add_landmarks(np.zeros(126, dtype=np.float32))
        self.engine.reset()
        self.assertEqual(self.engine.sequence_count, 0)
        self.assertFalse(self.engine.is_ready())

    def test_prediction_before_ready_raises_runtime_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.engine.predict()

    def test_prediction_with_synthetic_sequence(self) -> None:
        synthetic_sequence = np.linspace(0.0, 1.0, 32 * 126, dtype=np.float32).reshape(32, 126)
        for frame in synthetic_sequence:
            self.engine.add_landmarks(frame)

        result = self.engine.predict()
        self._assert_valid_prediction(result)

    def test_confidence_threshold_returns_unknown(self) -> None:
        threshold_engine = InferenceEngine(
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
            confidence_threshold=1.1,
        )
        threshold_engine.load()
        for frame in self.real_sequence:
            threshold_engine.add_landmarks(frame)

        result = threshold_engine.predict()
        self._assert_valid_prediction(result, threshold_engine)
        self.assertEqual(result["predicted_class"], "unknown")

    def test_real_silentvoice_dataset_integration(self) -> None:
        for frame in self.real_sequence:
            self.engine.add_landmarks(frame)

        result = self.engine.predict()
        self._assert_valid_prediction(result)
        self.assertIn(self.real_label, self.engine.index_to_class)

    def _assert_valid_prediction(
        self, result: dict[str, object], engine: InferenceEngine | None = None
    ) -> None:
        engine = engine or self.engine
        self.assertIn("predicted_index", result)
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIsInstance(result["predicted_index"], int)
        self.assertIsInstance(result["predicted_class"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertIsInstance(result["probabilities"], list)
        self.assertEqual(len(result["probabilities"]), engine.num_classes)
        self.assertTrue(all(isinstance(value, float) for value in result["probabilities"]))


if __name__ == "__main__":
    unittest.main()
