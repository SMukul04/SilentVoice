"""Test script for validating the LSTMModelBuilder class and the constructed model."""

import os
import sys
import numpy as np

# Ensure root of the workspace is in sys.path if not run with python -m
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.models.lstm_model import LSTMModelBuilder


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
    """Runs all 8 verification tests for the LSTMModelBuilder class."""
    results = {}

    print("===================================")
    print("LSTM MODEL TEST")
    print("===================================")

    # Test 1: Component initialization
    try:
        builder = LSTMModelBuilder(
            input_shape=(32, 126),
            num_classes=13,
            lstm_units=(64, 64),
            dropout_rate=0.5,
            dense_units=64,
            learning_rate=0.001,
        )
        assert builder.input_shape == (32, 126)
        assert builder.num_classes == 13
        assert builder.lstm_units == (64, 64)
        assert builder.dropout_rate == 0.5
        assert builder.dense_units == 64
        assert builder.learning_rate == 0.001
        results["Test 1: Component initialization"] = "PASSED"
    except Exception as e:
        results["Test 1: Component initialization"] = f"FAILED: {e}"

    # Test 2: Model construction
    try:
        builder = LSTMModelBuilder()
        model = builder.build_model()
        assert model is not None
        results["Test 2: Model construction"] = "PASSED"
    except Exception as e:
        results["Test 2: Model construction"] = f"FAILED: {e}"

    # Test 3: Model input shape
    try:
        builder = LSTMModelBuilder(input_shape=(32, 126))
        model = builder.build_model()
        assert model.input_shape == (None, 32, 126), f"Expected input shape (None, 32, 126), got {model.input_shape}"
        results["Test 3: Model input shape"] = "PASSED"
    except Exception as e:
        results["Test 3: Model input shape"] = f"FAILED: {e}"

    # Test 4: Model output shape
    try:
        builder = LSTMModelBuilder(num_classes=13)
        model = builder.build_model()
        assert model.output_shape == (None, 13), f"Expected output shape (None, 13), got {model.output_shape}"
        results["Test 4: Model output shape"] = "PASSED"
    except Exception as e:
        results["Test 4: Model output shape"] = f"FAILED: {e}"

    # Test 5: Model compilation
    try:
        builder = LSTMModelBuilder()
        model = builder.compile_model()
        assert model.optimizer is not None
        # Check compiled loss
        loss = getattr(model, "loss", None)
        assert loss == "sparse_categorical_crossentropy", f"Expected loss sparse_categorical_crossentropy, got {loss}"
        results["Test 5: Model compilation"] = "PASSED"
    except Exception as e:
        results["Test 5: Model compilation"] = f"FAILED: {e}"

    # Test 6: Forward pass with synthetic data
    try:
        batch_size = 4
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        model = builder.get_model()

        # Generate synthetic input
        synthetic_input = np.random.rand(batch_size, 32, 126).astype(np.float32)

        # Execute forward pass
        predictions = model(synthetic_input, training=False).numpy()

        # Verify output properties
        assert predictions.shape == (batch_size, 13), f"Expected predictions shape {(batch_size, 13)}, got {predictions.shape}"
        assert not np.any(np.isnan(predictions)), "Forward pass output contains NaNs"
        assert np.all(predictions >= 0.0), "Forward pass output contains negative values"
        assert np.all(predictions <= 1.0), "Forward pass output contains values greater than 1"
        assert np.allclose(np.sum(predictions, axis=-1), 1.0, atol=1e-5), "Softmax probabilities do not sum to 1"

        results["Test 6: Forward pass with synthetic data"] = "PASSED"
    except Exception as e:
        results["Test 6: Forward pass with synthetic data"] = f"FAILED: {e}"

    # Test 7: Invalid parameter handling
    try:
        # Invalid input_shape type
        assert check_exception(LSTMModelBuilder, (TypeError, ValueError), input_shape="invalid")
        # Invalid input_shape dims (must be 2D)
        assert check_exception(LSTMModelBuilder, (TypeError, ValueError), input_shape=(32, 126, 1))
        # Sequence length <= 0
        assert check_exception(LSTMModelBuilder, ValueError, input_shape=(0, 126))
        assert check_exception(LSTMModelBuilder, ValueError, input_shape=(-5, 126))
        assert check_exception(LSTMModelBuilder, TypeError, input_shape=("32", 126))
        # Feature dimension <= 0
        assert check_exception(LSTMModelBuilder, ValueError, input_shape=(32, 0))
        assert check_exception(LSTMModelBuilder, ValueError, input_shape=(32, -10))
        assert check_exception(LSTMModelBuilder, TypeError, input_shape=(32, "126"))

        # num_classes <= 1
        assert check_exception(LSTMModelBuilder, ValueError, num_classes=1)
        assert check_exception(LSTMModelBuilder, ValueError, num_classes=0)
        assert check_exception(LSTMModelBuilder, TypeError, num_classes="13")

        # invalid lstm_units
        assert check_exception(LSTMModelBuilder, ValueError, lstm_units=-1)
        assert check_exception(LSTMModelBuilder, ValueError, lstm_units=(64, 0))
        assert check_exception(LSTMModelBuilder, TypeError, lstm_units="invalid")

        # invalid dense_units
        assert check_exception(LSTMModelBuilder, ValueError, dense_units=-10)
        assert check_exception(LSTMModelBuilder, ValueError, dense_units=0)
        assert check_exception(LSTMModelBuilder, TypeError, dense_units="64")

        # dropout_rate outside valid range
        assert check_exception(LSTMModelBuilder, ValueError, dropout_rate=-0.1)
        assert check_exception(LSTMModelBuilder, ValueError, dropout_rate=1.5)
        assert check_exception(LSTMModelBuilder, TypeError, dropout_rate="0.5")

        # learning_rate <= 0
        assert check_exception(LSTMModelBuilder, ValueError, learning_rate=-0.001)
        assert check_exception(LSTMModelBuilder, ValueError, learning_rate=0.0)
        assert check_exception(LSTMModelBuilder, TypeError, learning_rate="0.001")

        results["Test 7: Invalid parameter handling"] = "PASSED"
    except Exception as e:
        results["Test 7: Invalid parameter handling"] = f"FAILED: {e}"

    # Test 8: Real dataset integration
    try:
        dataset_path = "datasets/landmarks/train.npz"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Real dataset not found at {dataset_path}")

        # Load train landmarks
        dataset = np.load(dataset_path)
        assert "X" in dataset, "Dataset does not contain 'X' key"
        assert "y" in dataset, "Dataset does not contain 'y' key"

        X_train = dataset["X"]

        # Real dataset dimensions
        assert X_train.ndim == 3, f"Expected 3D real data array, got {X_train.ndim}D"
        assert X_train.shape[1:] == (32, 126), f"Expected sample shape (32, 126), got {X_train.shape[1:]}"

        # Initialize model builder
        builder = LSTMModelBuilder(input_shape=(32, 126), num_classes=13)
        model = builder.get_model()

        # Select a small batch from the real dataset
        batch_size = min(5, len(X_train))
        real_batch = X_train[:batch_size].astype(np.float32)

        # Run forward pass
        predictions = model(real_batch, training=False).numpy()

        # Verify predictions
        assert predictions.shape == (batch_size, 13), f"Expected predictions shape {(batch_size, 13)}, got {predictions.shape}"
        assert not np.any(np.isnan(predictions)), "Forward pass on real data contains NaNs"
        assert np.all(predictions >= 0.0), "Forward pass output contains negative values"
        assert np.all(predictions <= 1.0), "Forward pass output contains values greater than 1"
        assert np.allclose(np.sum(predictions, axis=-1), 1.0, atol=1e-5), "Softmax probabilities do not sum to 1"

        results["Test 8: Real dataset integration"] = "PASSED"
    except Exception as e:
        results["Test 8: Real dataset integration"] = f"FAILED: {e}"

    # Print results
    for test_name in [
        "Test 1: Component initialization",
        "Test 2: Model construction",
        "Test 3: Model input shape",
        "Test 4: Model output shape",
        "Test 5: Model compilation",
        "Test 6: Forward pass with synthetic data",
        "Test 7: Invalid parameter handling",
        "Test 8: Real dataset integration",
    ]:
        status = results.get(test_name, "NOT RUN")
        print(f"{test_name}")
        print(f"{status}\n")

    any_failed = any("FAILED" in str(status) for status in results.values())
    if any_failed:
        print("Some LSTMModelBuilder tests FAILED!")
        sys.exit(1)
    else:
        print("All LSTMModelBuilder tests completed successfully!")


if __name__ == "__main__":
    main()
