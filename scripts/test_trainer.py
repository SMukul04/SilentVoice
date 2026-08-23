"""Test script for validating the ModelTrainer class and training orchestration."""

import json
import os
from pathlib import Path
import sys
import numpy as np

# Ensure root of the workspace is in sys.path if not run with python -m
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.models.lstm_model import LSTMModelBuilder
from backend.training.trainer import ModelTrainer


def check_exception(func, expected_exception, *args, **kwargs) -> bool:
    """Helper to verify if a function raises an expected exception."""
    try:
        func(*args, **kwargs)
        return False
    except expected_exception:
        return True
    except Exception:
        return False


def main() -> None:
    """Runs all 9 verification tests for the ModelTrainer class."""
    results = {}

    print("===================================")
    print("MODEL TRAINER TEST")
    print("===================================")

    # Establish scratch directories for test outputs
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)

    test_train_path = scratch_dir / "test_train.npz"
    test_val_path = scratch_dir / "test_val.npz"
    test_model_path = scratch_dir / "test_model.keras"
    test_history_path = scratch_dir / "test_history.json"
    integration_model_path = scratch_dir / "integration_model.keras"
    integration_history_path = scratch_dir / "integration_history.json"

    # Setup dummy data for loading/training tests
    dummy_X_train = np.random.rand(10, 32, 126).astype(np.float32)
    dummy_y_train = np.random.randint(0, 13, size=(10,)).astype(np.int32)
    dummy_X_val = np.random.rand(4, 32, 126).astype(np.float32)
    dummy_y_val = np.random.randint(0, 13, size=(4,)).astype(np.int32)

    np.savez(test_train_path, X=dummy_X_train, y=dummy_y_train)
    np.savez(test_val_path, X=dummy_X_val, y=dummy_y_val)

    # Test 1: Component initialization
    try:
        builder = LSTMModelBuilder()
        trainer = ModelTrainer(
            model_builder=builder,
            batch_size=16,
            epochs=2,
            patience=1,
            model_output_path=test_model_path,
            history_output_path=test_history_path,
            verbose=0,
        )
        assert trainer.model_builder is builder
        assert trainer.batch_size == 16
        assert trainer.epochs == 2
        assert trainer.patience == 1
        assert trainer.model_output_path == test_model_path
        assert trainer.history_output_path == test_history_path
        assert trainer.verbose == 0
        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # Test 2: Invalid parameter handling
    try:
        builder = LSTMModelBuilder()
        # Invalid builder
        assert check_exception(ModelTrainer, TypeError, model_builder=None)
        assert check_exception(ModelTrainer, TypeError, model_builder="invalid_builder")

        # Invalid batch_size
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, batch_size=0)
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, batch_size=-10)
        assert check_exception(ModelTrainer, TypeError, model_builder=builder, batch_size="32")

        # Invalid epochs
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, epochs=0)
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, epochs=-5)
        assert check_exception(ModelTrainer, TypeError, model_builder=builder, epochs="50")

        # Invalid patience
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, patience=-1)
        assert check_exception(ModelTrainer, TypeError, model_builder=builder, patience="5")

        # Invalid paths
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, model_output_path=" ")
        assert check_exception(ModelTrainer, TypeError, model_builder=builder, model_output_path=True)
        assert check_exception(ModelTrainer, ValueError, model_builder=builder, history_output_path="")
        assert check_exception(ModelTrainer, TypeError, model_builder=builder, history_output_path=123)

        results["Test 2: Invalid parameter handling"] = "PASSED"
    except Exception as e:
        results["Test 2: Invalid parameter handling"] = f"FAILED: {e}"

    # Test 3: Dataset loading
    try:
        builder = LSTMModelBuilder()
        trainer = ModelTrainer(
            model_builder=builder,
            model_output_path=test_model_path,
            history_output_path=test_history_path,
            verbose=0,
        )
        trainer.load_data(test_train_path, test_val_path)
        assert trainer.X_train.shape == (10, 32, 126)
        assert trainer.y_train.shape == (10,)
        assert trainer.X_val.shape == (4, 32, 126)
        assert trainer.y_val.shape == (4,)
        results["Test 3: Dataset loading"] = "PASSED"
    except Exception as e:
        results["Test 3: Dataset loading"] = f"FAILED: {e}"

    # Test 4: Dataset validation
    try:
        builder = LSTMModelBuilder()
        trainer = ModelTrainer(model_builder=builder, verbose=0)

        # Mismatched X and y sizes
        bad_X = np.random.rand(5, 32, 126)
        bad_y = np.random.randint(0, 13, size=(4,))
        assert check_exception(trainer.validate_data, ValueError, bad_X, bad_y)

        # Invalid X dimensions (must be 3D)
        bad_X_2d = np.random.rand(5, 126)
        bad_y_ok = np.random.randint(0, 13, size=(5,))
        assert check_exception(trainer.validate_data, ValueError, bad_X_2d, bad_y_ok)

        # Invalid y dimensions (must be 1D)
        bad_X_ok = np.random.rand(5, 32, 126)
        bad_y_2d = np.random.randint(0, 13, size=(5, 1))
        assert check_exception(trainer.validate_data, ValueError, bad_X_ok, bad_y_2d)

        # Empty arrays
        empty_X = np.empty((0, 32, 126))
        empty_y = np.empty((0,))
        assert check_exception(trainer.validate_data, ValueError, empty_X, empty_y)

        # NaN values in X
        nan_X = np.random.rand(5, 32, 126)
        nan_X[0, 0, 0] = np.nan
        nan_y = np.random.randint(0, 13, size=(5,))
        assert check_exception(trainer.validate_data, ValueError, nan_X, nan_y)

        # NaN values in y (converted to float to host NaN)
        nan_y_arr = np.random.randint(0, 13, size=(5,)).astype(np.float32)
        nan_y_arr[0] = np.nan
        assert check_exception(trainer.validate_data, ValueError, dummy_X_train[:5], nan_y_arr)

        results["Test 4: Dataset validation"] = "PASSED"
    except Exception as e:
        results["Test 4: Dataset validation"] = f"FAILED: {e}"

    # Test 5: Model building
    try:
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        trainer = ModelTrainer(model_builder=builder, verbose=0)
        model = trainer.build_model()
        assert model is not None
        assert model.input_shape == (None, 32, 126)
        assert model.output_shape == (None, 13)
        results["Test 5: Model building"] = "PASSED"
    except Exception as e:
        results["Test 5: Model building"] = f"FAILED: {e}"

    # Test 6: Short training run
    try:
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        trainer = ModelTrainer(
            model_builder=builder,
            batch_size=4,
            epochs=2,
            patience=1,
            model_output_path=test_model_path,
            history_output_path=test_history_path,
            verbose=0,
        )
        trainer.load_data(test_train_path, test_val_path)
        trainer.build_model()
        history = trainer.train()

        assert history is not None
        assert "loss" in history.history
        assert "val_loss" in history.history
        results["Test 6: Short training run"] = "PASSED"
    except Exception as e:
        results["Test 6: Short training run"] = f"FAILED: {e}"

    # Test 7: Model saving
    try:
        # Delete if already exists
        if test_model_path.exists():
            test_model_path.unlink()

        trainer.save_model()
        assert test_model_path.exists(), "Model file was not created"
        results["Test 7: Model saving"] = "PASSED"
    except Exception as e:
        results["Test 7: Model saving"] = f"FAILED: {e}"

    # Test 8: Training history saving
    try:
        # Delete if already exists
        if test_history_path.exists():
            test_history_path.unlink()

        trainer.save_history()
        assert test_history_path.exists(), "History JSON file was not created"

        # Try to parse written JSON
        with open(test_history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "loss" in data
        assert "val_loss" in data
        results["Test 8: Training history saving"] = "PASSED"
    except Exception as e:
        results["Test 8: Training history saving"] = f"FAILED: {e}"

    # Test 9: Complete pipeline integration (real dataset)
    try:
        real_train_path = Path("datasets/landmarks/train.npz")
        real_val_path = Path("datasets/landmarks/validation.npz")

        if not real_train_path.exists() or not real_val_path.exists():
            raise FileNotFoundError("Real train/validation datasets are missing!")

        if integration_model_path.exists():
            integration_model_path.unlink()
        if integration_history_path.exists():
            integration_history_path.unlink()

        # Build integration trainer
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        trainer = ModelTrainer(
            model_builder=builder,
            batch_size=8,
            epochs=1,
            patience=1,
            model_output_path=integration_model_path,
            history_output_path=integration_history_path,
            verbose=0,
        )

        # Run complete pipeline
        model, history = trainer.run(real_train_path, real_val_path)

        assert model is not None
        assert "loss" in history
        assert "val_loss" in history
        assert integration_model_path.exists(), "Integration model file not found"
        assert integration_history_path.exists(), "Integration history file not found"

        results["Test 9: Complete pipeline integration"] = "PASSED"
    except Exception as e:
        results["Test 9: Complete pipeline integration"] = f"FAILED: {e}"

    # Clean up mock files
    for path in [test_train_path, test_val_path, test_model_path, test_history_path, integration_model_path, integration_history_path]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    # Print results summary
    print()
    for test_name in [
        "Test 1: Component initialization",
        "Test 2: Invalid parameter handling",
        "Test 3: Dataset loading",
        "Test 4: Dataset validation",
        "Test 5: Model building",
        "Test 6: Short training run",
        "Test 7: Model saving",
        "Test 8: Training history saving",
        "Test 9: Complete pipeline integration",
    ]:
        status = results.get(test_name, "NOT RUN")
        print(f"{test_name}")
        print(f"{status}\n")

    any_failed = any("FAILED" in str(status) for status in results.values())
    if any_failed:
        print("Some ModelTrainer tests FAILED!")
        sys.exit(1)
    else:
        print("All ModelTrainer tests completed successfully!")


if __name__ == "__main__":
    main()
