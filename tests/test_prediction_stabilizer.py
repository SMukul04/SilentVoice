"""Tests for PredictionStabilizer, including real InferenceEngine integration."""

from pathlib import Path
import unittest
import numpy as np

from backend.realtime.prediction_stabilizer import PredictionStabilizer


def prediction(index: int = 0, label: str = "alive", probabilities: object = (0.8, 0.2), confidence: float = 0.8) -> dict:
    return {"predicted_index": index, "predicted_class": label, "confidence": confidence, "probabilities": probabilities}


class TestPredictionStabilizer(unittest.TestCase):
    def test_initialization(self):
        stabilizer = PredictionStabilizer()
        self.assertEqual(stabilizer.history_size, 0)
        self.assertFalse(stabilizer.is_stable())

    def test_invalid_configuration(self):
        for args in ((0, 0.0, 1), (1, 1.1, 1), (1, 0.0, 0)):
            with self.assertRaises((TypeError, ValueError)):
                PredictionStabilizer(*args)

    def test_invalid_prediction_input(self):
        stabilizer = PredictionStabilizer()
        invalid = (None, [], {}, prediction(probabilities=[]), prediction(probabilities=[[.5, .5]]), prediction(probabilities=["x"]), prediction(probabilities=[np.nan]), prediction(probabilities=[np.inf]), prediction(probabilities=[-.1, 1.1]), prediction(index=3), prediction(confidence=float("nan")), prediction(label=""))
        for item in invalid:
            with self.subTest(item=repr(item)[:40]):
                with self.assertRaises((TypeError, ValueError)):
                    stabilizer.add_prediction(item)

    def test_single_prediction(self):
        result = PredictionStabilizer().add_prediction(prediction())
        self.assertEqual(result["predicted_class"], "alive")

    def test_probability_averaging(self):
        s = PredictionStabilizer(window_size=2)
        s.add_prediction(prediction(probabilities=[.8, .2]))
        result = s.add_prediction(prediction(index=1, label="clean", probabilities=[.2, .8], confidence=.8))
        self.assertEqual(result["probabilities"], [.5, .5])

    def test_window_size(self):
        s = PredictionStabilizer(window_size=2)
        s.add_prediction(prediction(probabilities=[1., 0.]))
        s.add_prediction(prediction(probabilities=[1., 0.]))
        result = s.add_prediction(prediction(index=1, label="clean", probabilities=[0., 1.], confidence=1.))
        self.assertEqual(s.history_size, 2)
        self.assertEqual(result["probabilities"], [.5, .5])

    def test_consistency(self):
        s = PredictionStabilizer(min_consistent_predictions=3)
        self.assertEqual(s.add_prediction(prediction())["predicted_class"], "unknown")
        self.assertEqual(s.add_prediction(prediction())["predicted_class"], "unknown")
        self.assertEqual(s.add_prediction(prediction())["predicted_class"], "alive")

    def test_confidence_threshold(self):
        result = PredictionStabilizer(confidence_threshold=.9).add_prediction(prediction(confidence=.8))
        self.assertEqual(result["predicted_class"], "unknown")

    def test_incompatible_lengths(self):
        s = PredictionStabilizer()
        s.add_prediction(prediction(probabilities=[.5, .5]))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            s.add_prediction(prediction(probabilities=[.3, .3, .4]))

    def test_reset(self):
        s = PredictionStabilizer(); s.add_prediction(prediction()); s.reset()
        self.assertEqual(s.history_size, 0); self.assertFalse(s.is_stable())

    def test_json_compatibility(self):
        result = PredictionStabilizer().add_prediction(prediction())
        self.assertIsInstance(result["predicted_index"], int); self.assertIsInstance(result["predicted_class"], str)
        self.assertIsInstance(result["confidence"], float); self.assertIsInstance(result["probabilities"], list)

    def test_inference_engine_integration(self):
        from backend.inference.inference_engine import InferenceEngine
        root = Path(__file__).resolve().parents[1]
        engine = InferenceEngine(root / "models/checkpoints/best_model.keras", root / "datasets/landmarks/metadata.json")
        engine.load()
        with np.load(root / "datasets/landmarks/test.npz") as data:
            for frame in data["X"][0]: engine.add_landmarks(frame)
        result = PredictionStabilizer().add_prediction(engine.predict())
        self.assertEqual(set(result), {"predicted_index", "predicted_class", "confidence", "probabilities"})

