"""Optional Keras LSTM model and sequence-preparation helper."""

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_lstm_sequences(
    features: pd.DataFrame | np.ndarray,
    target: Sequence[float] | np.ndarray | pd.Series,
    reference_prices: Sequence[float] | np.ndarray | pd.Series,
    sequence_length: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn aligned tabular rows into rolling LSTM inputs and targets."""
    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(target, dtype=float)
    reference_values = np.asarray(reference_prices, dtype=float)

    if feature_values.ndim != 2:
        raise ValueError("features must be a two-dimensional array or DataFrame.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if not (
        len(feature_values) == len(target_values) == len(reference_values)
    ):
        raise ValueError("features, target, and reference_prices must have equal lengths.")
    if len(feature_values) < sequence_length:
        raise ValueError("There are fewer rows than sequence_length.")
    if not (
        np.isfinite(feature_values).all()
        and np.isfinite(target_values).all()
        and np.isfinite(reference_values).all()
    ):
        raise ValueError("Sequence inputs must contain only finite values.")

    sequences = np.stack(
        [
            feature_values[end - sequence_length + 1 : end + 1]
            for end in range(sequence_length - 1, len(feature_values))
        ]
    )
    return (
        sequences,
        target_values[sequence_length - 1 :],
        reference_values[sequence_length - 1 :],
    )


class LSTMModel:
    """Two-layer LSTM (64/32 units) with dropout and early stopping.

    Fit and predict inputs must be 3D arrays shaped as
    (samples, timesteps, features); use :func:`make_lstm_sequences` to create
    them from the tabular dataset.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 0,
    ) -> None:
        if epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError("epochs, batch_size, and patience must be at least 1.")
        try:
            keras = import_module("tensorflow").keras
        except ImportError as error:
            raise ImportError(
                "LSTMModel requires the optional 'tensorflow' package."
            ) from error

        self._keras: Any = keras
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.model: Any = None

    def fit(self, features: np.ndarray, target: Sequence[float] | np.ndarray) -> "LSTMModel":
        feature_values = self._validate_features(features)
        target_values = np.asarray(target, dtype=float)
        if len(feature_values) != len(target_values):
            raise ValueError("features and target must have equal lengths.")
        if len(feature_values) < 3:
            raise ValueError("At least three sequences are required to train the LSTM.")
        if not np.isfinite(target_values).all():
            raise ValueError("target must contain only finite values.")

        sample_count, timestep_count, feature_count = feature_values.shape
        scaled_features = self.scaler.fit_transform(
            feature_values.reshape(-1, feature_count)
        ).reshape(sample_count, timestep_count, feature_count)

        self.model = self._keras.Sequential(
            [
                self._keras.layers.Input(shape=(timestep_count, feature_count)),
                self._keras.layers.LSTM(64, return_sequences=True),
                self._keras.layers.Dropout(0.2),
                self._keras.layers.LSTM(32),
                self._keras.layers.Dropout(0.2),
                self._keras.layers.Dense(16, activation="relu"),
                self._keras.layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        early_stopping = self._keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        self.model.fit(
            scaled_features,
            target_values,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=self.verbose,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The LSTMModel must be fitted before calling predict.")
        feature_values = self._validate_features(features)
        sample_count, timestep_count, feature_count = feature_values.shape
        scaled_features = self.scaler.transform(
            feature_values.reshape(-1, feature_count)
        ).reshape(sample_count, timestep_count, feature_count)
        return self.model.predict(scaled_features, verbose=0).reshape(-1)

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        feature_values = np.asarray(features, dtype=float)
        if feature_values.ndim != 3:
            raise ValueError(
                "LSTM features must have shape (samples, timesteps, features)."
            )
        if not np.isfinite(feature_values).all():
            raise ValueError("LSTM features must contain only finite values.")
        return feature_values
