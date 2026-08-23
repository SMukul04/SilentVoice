"""Verification test script for validating the ModelEvaluator class."""

import json
import os
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.evaluation.model_evaluator import ModelEvaluator


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
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)

    test_model_path = scratch_dir / "temp_mock_model.keras"
    test_data_path = scratch_dir / "temp_test_data.npz"
    test_meta_path = scratch_dir / "temp_metadata.json"
    test_output_dir = scratch_dir / "eval_outputs"

    # Clean folders
    if test_output_dir.exists():
        import shutil
        shutil.rmtree(test_output_dir)

    # Mock metadata file
    mock_classes = {
        "num_classes": 13,
        "class_to_index": {
            "alive": 0, "clean": 1, "dead": 2, "deep": 3, "dirty": 4,
            "hard": 5, "heavy": 6, "high": 7, "low": 8, "shallow": 9,
            "soft": 10, "strong": 11, "weak": 12
        },
        "index_to_class": {
            "0": "alive", "1": "clean", "2": "dead", "3": "deep", "4": "dirty",
            "5": "hard", "6": "heavy", "7": "high", "8": "low", "9": "shallow",
            "10": "soft", "11": "strong", "12": "weak"
        }
    }
    with open(test_meta_path, "w", encoding="utf-8") as f:
        json.dump(mock_classes, f)

    # Mock test data file
    mock_X = np.random.rand(10, 32, 126).astype(np.float32)
    mock_y = np.random.randint(0, 13, size=(10,)).astype(np.int32)
    np.savez(test_data_path, X=mock_X, y=mock_y)

    # Mock Keras model
    inputs = tf.keras.Input(shape=(32, 126))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(13, activation="softmax")(x)
    mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    mock_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    mock_model.save(str(test_model_path))

    # Test 1: Component initialization
    try:
        evaluator = ModelEvaluator(
            model_path=test_model_path,
            test_data_path=test_data_path,
            metadata_path=test_meta_path,
            output_dir=test_output_dir,
        )
        assert evaluator.model_path == test_model_path
        assert evaluator.test_data_path == test_data_path
        assert evaluator.metadata_path == test_meta_path
        assert evaluator.output_dir == test_output_dir
        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # Test 2: Missing file handling
    try:
        missing_eval = ModelEvaluator(
            model_path=scratch_dir / "non_existent_model.keras",
            test_data_path=test_data_path,
            metadata_path=test_meta_path,
        )
        assert check_exception(missing_eval.load_model, FileNotFoundError)
        results["Test 2: Missing file handling"] = "PASSED"
    except Exception as e:
        results["Test 2: Missing file handling"] = f"FAILED: {e}"

    # Test 3: Synthetic dataset loading
    try:
        evaluator = ModelEvaluator(
            model_path=test_model_path,
            test_data_path=test_data_path,
            metadata_path=test_meta_path,
        )
        evaluator.load_test_dataset()
        assert evaluator.X_test.shape == (10, 32, 126)
        assert evaluator.y_test.shape == (10,)
        results["Test 3: Synthetic dataset loading"] = "PASSED"
    except Exception as e:
        results["Test 3: Synthetic dataset loading"] = f"FAILED: {e}"

    # Test 4: Dataset validation
    try:
        evaluator = ModelEvaluator(metadata_path=test_meta_path)
        bad_test_path = scratch_dir / "temp_bad_dataset.npz"

        def verify_validation_error(X, y, error_message_part):
            if X is not None and y is not None:
                np.savez(bad_test_path, X=X, y=y)
            elif X is not None:
                np.savez(bad_test_path, X=X)
            elif y is not None:
                np.savez(bad_test_path, y=y)
            else:
                with open(bad_test_path, "w") as f:
                    f.write("corrupted data")

            evaluator.test_data_path = bad_test_path
            try:
                evaluator.load_test_dataset()
                raise AssertionError(f"Dataset validation failed: Expected ValueError for {error_message_part}")
            except ValueError as ve:
                assert len(str(ve).strip()) > 0, f"Error message must not be empty for {error_message_part}"
            except Exception as ex:
                raise AssertionError(
                    f"Dataset validation failed: Expected ValueError for {error_message_part}, got {type(ex).__name__}: {ex}"
                )

        # 4.1: Missing X handling
        verify_validation_error(X=None, y=np.array([1, 2]), error_message_part="Missing X array")

        # 4.2: Missing y handling
        verify_validation_error(X=np.random.rand(2, 32, 126), y=None, error_message_part="Missing y array")

        # 4.3: Empty dataset handling
        verify_validation_error(X=np.empty((0, 32, 126)), y=np.empty((0,)), error_message_part="Empty dataset")

        # 4.4: Mismatched X/y lengths
        verify_validation_error(X=np.random.rand(5, 32, 126), y=np.array([1, 2, 3, 4]), error_message_part="mismatched X/y lengths")

        # 4.5: Invalid X dimensions (not 3D)
        verify_validation_error(X=np.random.rand(5, 126), y=np.random.randint(0, 13, size=(5,)), error_message_part="X must be 3-dimensional")

        # 4.6: Invalid y dimensions (not 1D)
        verify_validation_error(X=np.random.rand(5, 32, 126), y=np.random.randint(0, 13, size=(5, 1)), error_message_part="y must be 1-dimensional")

        # 4.7: NaN values check
        nan_X = np.random.rand(5, 32, 126)
        nan_X[0, 0, 0] = np.nan
        verify_validation_error(X=nan_X, y=np.random.randint(0, 13, size=(5,)), error_message_part="contains NaN values")

        # 4.8: Invalid or corrupted NPZ data
        verify_validation_error(X=None, y=None, error_message_part="Invalid or corrupted .npz")

        if bad_test_path.exists():
            bad_test_path.unlink()

        results["Test 4: Dataset validation"] = "PASSED"
    except Exception as e:
        results["Test 4: Dataset validation"] = f"FAILED: {e}"

    # Setup loaded evaluator for analytical tests
    evaluator = ModelEvaluator(
        model_path=test_model_path,
        test_data_path=test_data_path,
        metadata_path=test_meta_path,
        output_dir=test_output_dir,
    )
    evaluator.load_model()
    evaluator.load_test_dataset()
    evaluator.load_metadata()

    # Test 5: Prediction generation using lightweight mock model
    try:
        probs, preds, true_inds = evaluator.generate_predictions()
        assert probs.shape == (10, 13)
        assert preds.shape == (10,)
        assert true_inds.shape == (10,)
        results["Test 5: Prediction generation using a lightweight mock or synthetic model"] = "PASSED"
    except Exception as e:
        results["Test 5: Prediction generation using a lightweight mock or synthetic model"] = f"FAILED: {e}"

    # Test 6: Classification report generation
    try:
        _, preds, _ = evaluator.generate_predictions()
        report = evaluator.generate_classification_report(preds)
        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "alive" in report
        results["Test 6: Classification report generation"] = "PASSED"
    except Exception as e:
        results["Test 6: Classification report generation"] = f"FAILED: {e}"

    # Test 7: Confusion matrix generation
    try:
        _, preds, _ = evaluator.generate_predictions()
        cm = evaluator.generate_confusion_matrix(preds)
        assert cm.shape == (13, 13)
        results["Test 7: Confusion matrix generation"] = "PASSED"
    except Exception as e:
        results["Test 7: Confusion matrix generation"] = f"FAILED: {e}"

    # Test 8: Per-class accuracy calculation
    try:
        _, preds, _ = evaluator.generate_predictions()
        per_class = evaluator.calculate_per_class_accuracy(preds)
        assert len(per_class) == 13
        assert "alive" in per_class
        results["Test 8: Per-class accuracy calculation"] = "PASSED"
    except Exception as e:
        results["Test 8: Per-class accuracy calculation"] = f"FAILED: {e}"

    # Test 9: Saving evaluation artifacts
    try:
        loss, acc = evaluator.evaluate_loss_and_accuracy()
        _, preds, _ = evaluator.generate_predictions()
        report = evaluator.generate_classification_report(preds)
        cm = evaluator.generate_confusion_matrix(preds)
        per_class = evaluator.calculate_per_class_accuracy(preds)

        evaluator.save_results(loss, acc, per_class, report, cm)
        
        assert (test_output_dir / "evaluation_results.json").exists()
        assert (test_output_dir / "classification_report.json").exists()
        assert (test_output_dir / "confusion_matrix.npy").exists()

        results["Test 9: Saving evaluation artifacts"] = "PASSED"
    except Exception as e:
        results["Test 9: Saving evaluation artifacts"] = f"FAILED: {e}"

    # Test 10: Real SilentVoice integration
    real_model_path = "models/checkpoints/best_model.keras"
    real_test_path = "datasets/landmarks/test.npz"
    real_meta_path = "datasets/landmarks/metadata.json"
    real_output_dir = Path("artifacts/evaluation")

    real_results = None
    real_integration_status = "FAILED"

    try:
        if os.path.exists(real_model_path) and os.path.exists(real_test_path) and os.path.exists(real_meta_path):
            real_evaluator = ModelEvaluator(
                model_path=real_model_path,
                test_data_path=real_test_path,
                metadata_path=real_meta_path,
                output_dir=real_output_dir,
            )
            real_results = real_evaluator.evaluate()
            real_integration_status = "PASSED"
            results["Test 10: Real SilentVoice integration"] = "PASSED"
        else:
            results["Test 10: Real SilentVoice integration"] = "FAILED: Real files missing"
    except Exception as e:
        results["Test 10: Real SilentVoice integration"] = f"FAILED: {e}"

    # Clean up mock files
    for path in [test_model_path, test_data_path, test_meta_path]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    import shutil
    if test_output_dir.exists():
        try:
            shutil.rmtree(test_output_dir)
        except OSError:
            pass

    # Print results summary
    print("\n===================================")
    print("MODEL EVALUATOR TEST")
    print("===================================\n")

    for i in range(1, 11):
        test_name = f"Test {i}:"
        matching_key = next((k for k in results if k.startswith(test_name)), None)
        if matching_key:
            print(f"{matching_key}")
            print(f"{results[matching_key]}\n")
        else:
            print(f"Test {i}: Not found")
            print("FAILED\n")

    # Print real report summary
    print("===================================")
    print("REAL MODEL EVALUATION REPORT")
    print("===================================\n")
    if real_results:
        print(f"Test Samples: {real_results['total_test_samples']}")
        print(f"Test Loss: {real_results['test_loss']:.4f}")
        print(f"Test Accuracy: {real_results['test_accuracy']:.4f}\n")
        print("Per-Class Accuracy:")
        for class_name, acc in real_results["per_class_accuracy"].items():
            print(f"{class_name}: {acc:.4f}")
        print("\nSaved Files:")
        print(f"- {real_output_dir}/evaluation_results.json")
        print(f"- {real_output_dir}/classification_report.json")
        print(f"- {real_output_dir}/confusion_matrix.npy\n")
    else:
        print("Real model evaluation was not performed or failed.\n")

    any_failed = any("FAILED" in str(status) for status in results.values())
    if any_failed:
        print("Some ModelEvaluator tests FAILED!")
        sys.exit(1)
    else:
        print("All ModelEvaluator tests completed successfully!")


if __name__ == "__main__":
    main()
