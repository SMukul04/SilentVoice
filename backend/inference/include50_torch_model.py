"""PyTorch model architecture for the INCLUDE-50 sign-language classifier."""

import torch.nn as nn


class SignLSTM(nn.Module):
    """Two-layer LSTM classifier matching the trained INCLUDE-50 model."""

    def __init__(
        self,
        input_features=126,
        hidden_units=64,
        dense_units=64,
        num_classes=50,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_units,
            hidden_size=hidden_units,
            batch_first=True,
        )

        self.dropout2 = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        x = self.dropout2(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout3(x)

        return self.output(x)
