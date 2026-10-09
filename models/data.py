"""Helpers for preparing time-ordered price-prediction datasets."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def build_supervised_dataset(
    data: pd.DataFrame,
    target_column: str = "Close",
    lag_count: int = 5,
    horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build next-period targets without including the target row as a feature.

    Returns feature rows (including the current target as a predictor), future
    target values, and the current target values used for directional scoring.
    """
    if data is None or data.empty:
        raise ValueError("data must be a non-empty DataFrame.")
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} is missing.")
    if lag_count < 1:
        raise ValueError("lag_count must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    ordered_data = data.sort_index()
    if feature_columns is None:
        selected_columns = [
            column
            for column in ordered_data.select_dtypes(include="number").columns
            if column != target_column
        ]
    else:
        selected_columns = list(feature_columns)
        if target_column in selected_columns:
            raise ValueError(
                f"{target_column!r} is added as a current-price feature automatically."
            )
        missing_columns = [
            column for column in selected_columns if column not in ordered_data.columns
        ]
        if missing_columns:
            raise ValueError(f"Feature columns are missing: {missing_columns}")

    feature_frame = ordered_data[selected_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    current_price_column = f"{target_column}_current"
    if current_price_column in feature_frame.columns:
        raise ValueError(
            f"Generated current-price column {current_price_column!r} already exists."
        )
    current_prices = pd.to_numeric(ordered_data[target_column], errors="coerce")
    feature_frame[current_price_column] = current_prices
    for lag in range(1, lag_count + 1):
        lag_column = f"{target_column}_lag_{lag}"
        if lag_column in feature_frame.columns:
            raise ValueError(f"Generated lag column {lag_column!r} already exists.")
        feature_frame[lag_column] = pd.to_numeric(
            current_prices
        ).shift(lag)

    target = current_prices.shift(-horizon)
    reference_prices = current_prices
    valid_rows = (
        feature_frame.notna().all(axis=1)
        & np.isfinite(feature_frame).all(axis=1)
        & target.notna()
        & np.isfinite(target)
        & reference_prices.notna()
        & np.isfinite(reference_prices)
    )

    return (
        feature_frame.loc[valid_rows].copy(),
        target.loc[valid_rows].rename(target_column),
        reference_prices.loc[valid_rows].rename(f"{target_column}_reference"),
    )


def evaluate_predictions(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
    reference_prices: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate directional accuracy (percent), R², and MAE."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference_prices, dtype=float)

    if actual_values.ndim != 1 or predicted_values.ndim != 1 or reference_values.ndim != 1:
        raise ValueError("actual, predicted, and reference_prices must be one-dimensional.")
    if not (
        len(actual_values) == len(predicted_values) == len(reference_values)
    ):
        raise ValueError("actual, predicted, and reference_prices must have equal lengths.")
    if len(actual_values) < 2:
        raise ValueError("At least two predictions are required to calculate R².")
    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all()
        and np.isfinite(reference_values).all()
    ):
        raise ValueError("Metric inputs must contain only finite values.")

    actual_direction = np.sign(actual_values - reference_values)
    predicted_direction = np.sign(predicted_values - reference_values)
    return {
        "directional_accuracy": float(
            np.mean(actual_direction == predicted_direction) * 100
        ),
        "r2": float(r2_score(actual_values, predicted_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
    }
