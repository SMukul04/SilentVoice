"""Unit tests for ModelEvaluator module."""

import json
from pathlib import Path
import shutil
import unittest
import numpy as np

from backend.evaluation.model_evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test suite for validating ModelEvaluator logic."""

    @classmethod
    def setUpClass(cls):
        cls.scratch_dir = Path("scratch/test_evaluator_unittest")
        cls.scratch_dir.mkdir(parents=True, exist_ok=True)

        cls.dummy_train_npz = cls.scratch_dir / "train.npz"
        cls.dummy_test_npz = cls.scratch_dir / "test.npz"
        cls.dummy_metadata_json = cls.scratch_dir / "metadata.json"

        # Create mock dataset
        X = np.random.rand(5, 32, 126).astype(np.float32)
        y = np.random.randint(0, 3, size=(5,)).astype(np.int32)
        np.savez(cls.dummy_test_npz, X=X, y=y)

        # Create mock metadata
        meta = {
            "num_classes": 3,
            "class_to_index": {"alive": 0, "clean": 1, "dead": 2},
            "index_to_class": {"0": "alive", "1": "clean", "2": "dead"},
        }
        with open(cls.dummy_metadata_json, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def tearDownClass(cls):
        if cls.scratch_dir.exists():
            shutil.rmtree(cls.scratch_dir)

    def test_initialization(self):
        """Tests that paths and variables initialize correctly."""
        evaluator = ModelEvaluator(
            model_path="models/checkpoints/best_model.keras",
            test_data_path=self.dummy_test_npz,
            metadata_path=self.dummy_metadata_json,
            output_dir=self.scratch_dir / "results",
        )
        self.assertEqual(evaluator.model_path, Path("models/checkpoints/best_model.keras"))
        self.assertEqual(evaluator.test_data_path, self.dummy_test_npz)
        self.assertEqual(evaluator.metadata_path, self.dummy_metadata_json)
        self.assertEqual(evaluator.output_dir, self.scratch_dir / "results")

    def test_missing_files_raising_errors(self):
        """Tests that missing files raise FileNotFoundError."""
        # Missing model
        evaluator = ModelEvaluator(
            model_path=self.scratch_dir / "missing_model.keras",
            test_data_path=self.dummy_test_npz,
            metadata_path=self.dummy_metadata_json,
        )
        with self.assertRaises(FileNotFoundError):
            evaluator.load_model()

        # Missing test dataset
        evaluator = ModelEvaluator(
            model_path=self.scratch_dir / "missing_model.keras",
            test_data_path=self.scratch_dir / "missing_test.npz",
            metadata_path=self.dummy_metadata_json,
        )
        with self.assertRaises(FileNotFoundError):
            evaluator.load_test_dataset()

        # Missing metadata JSON
        evaluator = ModelEvaluator(
            model_path=self.scratch_dir / "missing_model.keras",
            test_data_path=self.dummy_test_npz,
            metadata_path=self.scratch_dir / "missing_metadata.json",
        )
        with self.assertRaises(FileNotFoundError):
            evaluator.load_metadata()

    def test_dataset_validations(self):
        """Tests that invalid datasets raise ValueError during load."""
        evaluator = ModelEvaluator(metadata_path=self.dummy_metadata_json)

        # Mismatched lengths
        bad_test_file = self.scratch_dir / "bad_test.npz"
        np.savez(bad_test_file, X=np.random.rand(5, 32, 126), y=np.random.randint(0, 3, size=(4,)))
        evaluator.test_data_path = bad_test_file
        with self.assertRaises(ValueError):
            evaluator.load_test_dataset()

        # Let's verify dataset check directly
        bad_test_file = self.scratch_dir / "bad_test.npz"

        # NaN in features
        X_nan = np.random.rand(5, 32, 126).astype(np.float32)
        X_nan[0, 0, 0] = np.nan
        y_ok = np.random.randint(0, 3, size=(5,)).astype(np.int32)
        np.savez(bad_test_file, X=X_nan, y=y_ok)

        evaluator.test_data_path = bad_test_file
        with self.assertRaises(ValueError):
            evaluator.load_test_dataset()

        # Empty array
        X_empty = np.empty((0, 32, 126))
        y_empty = np.empty((0,))
        np.savez(bad_test_file, X=X_empty, y=y_empty)
        with self.assertRaises(ValueError):
            evaluator.load_test_dataset()

    def test_metadata_validations(self):
        """Tests that corrupted metadata JSON raises ValueError."""
        bad_meta_file = self.scratch_dir / "bad_metadata.json"
        
        # Missing keys
        bad_meta = {"num_classes": 3}
        with open(bad_meta_file, "w", encoding="utf-8") as f:
            json.dump(bad_meta, f)

        evaluator = ModelEvaluator(metadata_path=bad_meta_file)
        with self.assertRaises(ValueError):
            evaluator.load_metadata()

    def test_per_class_accuracy_calculation(self):
        """Tests class accuracy checks and zero-division handling."""
        evaluator = ModelEvaluator(
            test_data_path=self.dummy_test_npz,
            metadata_path=self.dummy_metadata_json,
        )
        evaluator.load_metadata()
        evaluator.y_test = np.array([0, 0, 1, 1, 2])
        preds = np.array([0, 1, 1, 1, 0])  # class 0: 1/2 correct, class 1: 2/2 correct, class 2: 0/1 correct

        per_class_acc = evaluator.calculate_per_class_accuracy(preds)
        self.assertEqual(per_class_acc["alive"], 0.5)
        self.assertEqual(per_class_acc["clean"], 1.0)
        self.assertEqual(per_class_acc["dead"], 0.0)


if __name__ == "__main__":
    unittest.main()
