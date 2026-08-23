"""Test script for verifying ProductionTrainer and resume-training pipeline functionality."""

import json
import os
from pathlib import Path
import sys
import time
import numpy as np

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.models.lstm_model import LSTMModelBuilder
from backend.training.production_trainer import ProductionTrainer


def check_exception(func, expected_exception, *args, **kwargs) -> bool:
    """Helper to verify if a function raises an expected exception."""
    try:
        func(*args, **kwargs)
        return False
    except expected_exception:
        return True
    except Exception:
        return False


def main():
    results = {}
    
    # Setup directories
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)
    
    test_train_path = scratch_dir / "prod_test_train.npz"
    test_val_path = scratch_dir / "prod_test_val.npz"
    test_checkpoint_dir = scratch_dir / "prod_checkpoints"
    test_history_path = scratch_dir / "prod_history.json"

    # Make clean folders
    if test_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(test_checkpoint_dir)
    test_checkpoint_dir.mkdir(exist_ok=True)

    # Setup dummy data for test datasets
    dummy_X_train = np.random.rand(10, 32, 126).astype(np.float32)
    dummy_y_train = np.random.randint(0, 13, size=(10,)).astype(np.int32)
    dummy_X_val = np.random.rand(4, 32, 126).astype(np.float32)
    dummy_y_val = np.random.randint(0, 13, size=(4,)).astype(np.int32)

    np.savez(test_train_path, X=dummy_X_train, y=dummy_y_train)
    np.savez(test_val_path, X=dummy_X_val, y=dummy_y_val)

    # Test 1: Component initialization
    try:
        builder = LSTMModelBuilder()
        trainer = ProductionTrainer(
            model_builder=builder,
            batch_size=16,
            epochs=2,
            patience=1,
            checkpoint_dir=test_checkpoint_dir,
            history_output_path=test_history_path,
            verbose=0,
        )
        assert trainer.model_builder is builder
        assert trainer.batch_size == 16
        assert trainer.epochs == 2
        assert trainer.patience == 1
        assert trainer.checkpoint_dir == test_checkpoint_dir
        assert trainer.history_output_path == test_history_path
        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # Test 2: Dataset loading
    try:
        builder = LSTMModelBuilder()
        trainer = ProductionTrainer(
            model_builder=builder,
            checkpoint_dir=test_checkpoint_dir,
            history_output_path=test_history_path,
            verbose=0,
        )
        trainer.load_datasets(test_train_path, test_val_path)
        assert trainer.X_train.shape == (10, 32, 126)
        assert trainer.y_train.shape == (10,)
        assert trainer.X_val.shape == (4, 32, 126)
        assert trainer.y_val.shape == (4,)
        results["Test 2: Dataset loading"] = "PASSED"
    except Exception as e:
        results["Test 2: Dataset loading"] = f"FAILED: {e}"

    # Test 3: Dataset validation
    try:
        builder = LSTMModelBuilder()
        trainer = ProductionTrainer(model_builder=builder, verbose=0)

        # Mismatched X and y sizes
        trainer.X_train = np.random.rand(5, 32, 126)
        trainer.y_train = np.random.randint(0, 13, size=(4,))
        trainer.X_val = dummy_X_val
        trainer.y_val = dummy_y_val
        assert check_exception(trainer.validate_datasets, ValueError)

        # Invalid X dimensions
        trainer.X_train = np.random.rand(5, 126)
        trainer.y_train = np.random.randint(0, 13, size=(5,))
        assert check_exception(trainer.validate_datasets, ValueError)

        # Invalid y dimensions
        trainer.X_train = dummy_X_train
        trainer.y_train = np.random.randint(0, 13, size=(10, 1))
        assert check_exception(trainer.validate_datasets, ValueError)

        # Empty arrays
        trainer.X_train = np.empty((0, 32, 126))
        trainer.y_train = np.empty((0,))
        assert check_exception(trainer.validate_datasets, ValueError)

        # NaN values in validation X
        nan_X = np.random.rand(4, 32, 126)
        nan_X[0, 0, 0] = np.nan
        trainer.X_train = dummy_X_train
        trainer.y_train = dummy_y_train
        trainer.X_val = nan_X
        trainer.y_val = dummy_y_val
        assert check_exception(trainer.validate_datasets, ValueError)

        results["Test 3: Dataset validation"] = "PASSED"
    except Exception as e:
        results["Test 3: Dataset validation"] = f"FAILED: {e}"

    # Test 4: Model building
    try:
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        trainer = ProductionTrainer(model_builder=builder, verbose=0)
        model = trainer.build_model()
        assert model is not None
        assert model.input_shape == (None, 32, 126)
        assert model.output_shape == (None, 13)
        results["Test 4: Model building"] = "PASSED"
    except Exception as e:
        results["Test 4: Model building"] = f"FAILED: {e}"

    # Setup files for training tests
    builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
    trainer = ProductionTrainer(
        model_builder=builder,
        batch_size=4,
        epochs=2,
        patience=1,
        checkpoint_dir=test_checkpoint_dir,
        history_output_path=test_history_path,
        verbose=0,
    )
    trainer.load_datasets(test_train_path, test_val_path)

    # Test 5: Fresh short training run
    try:
        model, history = trainer.train(resume=False)
        assert model is not None
        assert len(history.get("loss", [])) == 2
        results["Test 5: Fresh short training run"] = "PASSED"
    except Exception as e:
        results["Test 5: Fresh short training run"] = f"FAILED: {e}"

    # Test 6: Checkpoint creation
    try:
        latest_checkpoint = test_checkpoint_dir / "latest_model.keras"
        assert latest_checkpoint.exists(), "Latest model checkpoint not found"
        results["Test 6: Checkpoint creation"] = "PASSED"
    except Exception as e:
        results["Test 6: Checkpoint creation"] = f"FAILED: {e}"

    # Test 7: Best model creation
    try:
        best_checkpoint = test_checkpoint_dir / "best_model.keras"
        assert best_checkpoint.exists(), "Best model checkpoint not found"
        results["Test 7: Best model creation"] = "PASSED"
    except Exception as e:
        results["Test 7: Best model creation"] = f"FAILED: {e}"

    # Test 8: Training history saving
    try:
        assert test_history_path.exists(), "History JSON file not found"
        with open(test_history_path, "r", encoding="utf-8") as f:
            h_data = json.load(f)
        assert "loss" in h_data
        assert len(h_data["loss"]) == 2
        results["Test 8: Training history saving"] = "PASSED"
    except Exception as e:
        results["Test 8: Training history saving"] = f"FAILED: {e}"

    # Test 9: Training state saving
    try:
        state_file = test_checkpoint_dir / "training_state.json"
        assert state_file.exists(), "Training state JSON file not found"
        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        assert state_data["completed_epochs"] == 2
        assert state_data["total_epochs"] == 2
        results["Test 9: Training state saving"] = "PASSED"
    except Exception as e:
        results["Test 9: Training state saving"] = f"FAILED: {e}"

    # Test 10: Resume detection and resume training
    try:
        # Create a new trainer pointing to the same checkpoints, targeting 3 epochs
        resume_trainer = ProductionTrainer(
            model_builder=builder,
            batch_size=4,
            epochs=3,
            patience=1,
            checkpoint_dir=test_checkpoint_dir,
            history_output_path=test_history_path,
            verbose=0,
        )
        resume_trainer.load_datasets(test_train_path, test_val_path)
        
        # Verify resume checkpoint is detected
        assert resume_trainer.find_resume_checkpoint() is True, "Failed to detect checkpoint"
        
        # Run resume train
        model, history = resume_trainer.train(resume=True)
        
        # Verify it trained from epoch 2 to 3 (which yields 1 epoch index of fit metrics)
        assert len(history.get("loss", [])) == 1, f"Expected 1 resumed epoch, got {len(history.get('loss', []))}"
        
        # Check that state updated
        with open(test_checkpoint_dir / "training_state.json", "r", encoding="utf-8") as f:
            new_state = json.load(f)
        assert new_state["completed_epochs"] == 3
        
        results["Test 10: Resume detection and resume training"] = "PASSED"
    except Exception as e:
        results["Test 10: Resume detection and resume training"] = f"FAILED: {e}"

    # Real dataset integration test
    real_train_path = "datasets/landmarks/train.npz"
    real_val_path = "datasets/landmarks/validation.npz"
    integration_checkpoint_dir = scratch_dir / "prod_integration_checkpoints"
    integration_history_path = scratch_dir / "prod_integration_history.json"

    if integration_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(integration_checkpoint_dir)
    integration_checkpoint_dir.mkdir(exist_ok=True)

    real_data_loaded = False
    train_shape = "N/A"
    val_shape = "N/A"
    resume_support_status = "FAILED"

    try:
        if os.path.exists(real_train_path) and os.path.exists(real_val_path):
            # Run fresh 1-epoch run
            builder_real = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
            trainer_real = ProductionTrainer(
                model_builder=builder_real,
                batch_size=8,
                epochs=1,
                patience=1,
                checkpoint_dir=integration_checkpoint_dir,
                history_output_path=integration_history_path,
                verbose=0,
            )
            trainer_real.load_datasets(real_train_path, real_val_path)
            
            real_data_loaded = True
            train_shape = str(trainer_real.X_train.shape)
            val_shape = str(trainer_real.X_val.shape)

            # Fit 1 epoch
            trainer_real.train(resume=False)

            # Verify checkpoint is created
            assert (integration_checkpoint_dir / "latest_model.keras").exists()
            assert (integration_checkpoint_dir / "training_state.json").exists()

            # Resume integration run for epoch 2
            resume_real = ProductionTrainer(
                model_builder=builder_real,
                batch_size=8,
                epochs=2,
                patience=1,
                checkpoint_dir=integration_checkpoint_dir,
                history_output_path=integration_history_path,
                verbose=0,
            )
            resume_real.load_datasets(real_train_path, real_val_path)
            resume_real.train(resume=True)

            with open(integration_checkpoint_dir / "training_state.json", "r", encoding="utf-8") as f:
                int_state = json.load(f)
            assert int_state["completed_epochs"] == 2
            resume_support_status = "PASSED"
    except Exception as e:
        print(f"Error during real dataset integration: {e}")
        resume_support_status = f"FAILED: {e}"

    # Clean up mock files
    for path in [test_train_path, test_val_path, test_history_path, integration_history_path]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    import shutil
    for d in [test_checkpoint_dir, integration_checkpoint_dir]:
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError:
                pass

    # Print Report
    print("\n===================================")
    print("PRODUCTION TRAINER TEST")
    print("===================================\n")

    for i in range(1, 11):
        test_name = f"Test {i}"
        matching_key = next((k for k in results if k.startswith(test_name)), None)
        if matching_key:
            print(f"{matching_key}")
            print(f"{results[matching_key]}\n")
        else:
            print(f"Test {i}: Not found")
            print("FAILED\n")

    print("===================================")
    print("PRODUCTION TRAINING REPORT")
    print("===================================\n")
    print(f"Train Shape: {train_shape}")
    print(f"Validation Shape: {val_shape}")
    print(f"Checkpoint: models/checkpoints/latest_model.keras")
    print(f"Best Model: models/checkpoints/best_model.keras")
    print(f"Resume Support: {resume_support_status}\n")

    any_failed = any("FAILED" in str(status) for status in results.values()) or resume_support_status.startswith("FAILED")
    if any_failed:
        print("Some ProductionTrainer tests FAILED!")
        sys.exit(1)
    else:
        print("All ProductionTrainer tests completed successfully!")


if __name__ == "__main__":
    main()
