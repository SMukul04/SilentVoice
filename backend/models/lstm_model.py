"""LSTM Model Builder module for sign language sequence classification."""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class LSTMModelBuilder:
    """Builder class for constructing, compiling, and getting an LSTM model."""

    def __init__(
        self,
        input_shape=(32, 126),
        num_classes=13,
        lstm_units=(64, 64),
        dropout_rate=0.5,
        dense_units=64,
        learning_rate=0.001,
    ):
        """Initializes and validates parameters for the LSTM model.

        Args:
            input_shape (tuple): (sequence_length, feature_dimension). Defaults to (32, 126).
            num_classes (int): Number of target classes. Defaults to 13.
            lstm_units (int or sequence): LSTM units per layer. Defaults to (64, 64).
            dropout_rate (float): Dropout rate. Defaults to 0.5.
            dense_units (int): Number of dense units before output. Defaults to 64.
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.001.
        """
        # 1. Validate input_shape
        if not isinstance(input_shape, (tuple, list)):
            raise TypeError("input_shape must be a tuple or list")
        if len(input_shape) != 2:
            raise ValueError("input_shape must have exactly 2 dimensions (sequence_length, feature_dimension)")

        sequence_length, feature_dimension = input_shape

        # Check sequence length
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise TypeError("sequence_length must be an integer")
        if sequence_length <= 0:
            raise ValueError("sequence length <= 0")

        # Check feature dimension
        if not isinstance(feature_dimension, int) or isinstance(feature_dimension, bool):
            raise TypeError("feature_dimension must be an integer")
        if feature_dimension <= 0:
            raise ValueError("feature dimension <= 0")

        # 2. Validate num_classes
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError("num_classes must be an integer")
        if num_classes <= 1:
            raise ValueError("num_classes <= 1")

        # 3. Validate lstm_units
        if isinstance(lstm_units, (int, float)) and not isinstance(lstm_units, bool):
            if not isinstance(lstm_units, int):
                raise TypeError("lstm_units must be an integer or a tuple/list of integers")
            if lstm_units <= 0:
                raise ValueError("invalid lstm_units")
            units_list = (lstm_units, lstm_units)
        elif isinstance(lstm_units, (tuple, list)):
            if len(lstm_units) == 0:
                raise ValueError("invalid lstm_units")
            for u in lstm_units:
                if not isinstance(u, int) or isinstance(u, bool):
                    raise TypeError("Each element of lstm_units must be an integer")
                if u <= 0:
                    raise ValueError("invalid lstm_units")
            if len(lstm_units) == 1:
                units_list = (lstm_units[0], lstm_units[0])
            else:
                units_list = tuple(lstm_units)
        else:
            raise TypeError("lstm_units must be an integer or a tuple/list of integers")

        # 4. Validate dense_units
        if not isinstance(dense_units, int) or isinstance(dense_units, bool):
            raise TypeError("dense_units must be an integer")
        if dense_units <= 0:
            raise ValueError("invalid dense_units")

        # 5. Validate dropout_rate
        if not isinstance(dropout_rate, (int, float)) or isinstance(dropout_rate, bool):
            raise TypeError("dropout_rate must be a float or int")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate outside valid range")

        # 6. Validate learning_rate
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise TypeError("learning_rate must be a float or int")
        if learning_rate <= 0:
            raise ValueError("learning_rate <= 0")

        self.input_shape = (sequence_length, feature_dimension)
        self.num_classes = num_classes
        self.lstm_units = units_list
        self.dropout_rate = float(dropout_rate)
        self.dense_units = dense_units
        self.learning_rate = float(learning_rate)

        self.model = None

    def build_model(self):
        """Builds the LSTM model architecture.

        Returns:
            tf.keras.Model: Uncompiled Keras model.
        """
        inputs = layers.Input(shape=self.input_shape)

        # First LSTM layer (must return sequences because there is a second LSTM layer)
        x = layers.LSTM(self.lstm_units[0], return_sequences=True)(inputs)
        x = layers.Dropout(self.dropout_rate)(x)

        # Second LSTM layer (return_sequences=False because the next layer is Dense)
        x = layers.LSTM(self.lstm_units[1], return_sequences=False)(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Dense layer with ReLU
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output Dense layer with softmax
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_Sign_Classifier")
        return self.model

    def compile_model(self):
        """Compiles the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        if self.model is None:
            self.build_model()

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model

    def get_model(self):
        """Gets the compiled model, building and compiling it first if necessary.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        if self.model is None:
            self.build_model()

        # Check if the model has been compiled
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            self.compile_model()

        return self.model
