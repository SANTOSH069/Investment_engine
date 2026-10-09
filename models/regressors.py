"""Scikit-learn and XGBoost price-regression models."""

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class LinearRegressionModel(LinearRegression):
    """Interpretable linear-regression baseline."""


class RandomForestModel(RandomForestRegressor):
    """Random forest with 100 trees by default."""

    def __init__(self, n_estimators: int = 100) -> None:
        super().__init__(n_estimators=n_estimators, random_state=42)


class SVRModel:
    """RBF-kernel SVR with feature scaling."""

    def __init__(self, **kwargs: Any) -> None:
        self.estimator = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", **kwargs)),
            transformer=StandardScaler(),
        )

    def fit(
        self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray
    ) -> "SVRModel":
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(features))


class XGBoostModel:
    """XGBoost regressor with 200 boosting rounds and early stopping."""

    def __init__(
        self,
        n_estimators: int = 200,
        early_stopping_rounds: int = 20,
        validation_fraction: float = 0.2,
        **kwargs: Any,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be at least 1.")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")

        try:
            xgboost = import_module("xgboost")
        except ImportError as error:
            raise ImportError(
                "XGBoostModel requires the optional 'xgboost' package."
            ) from error

        self.validation_fraction = validation_fraction
        self.estimator: Any = xgboost.XGBRegressor(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            objective="reg:squarederror",
            eval_metric="mae",
            **kwargs,
        )

    def fit(
        self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray
    ) -> "XGBoostModel":
        if len(features) < 3:
            raise ValueError("At least three rows are required for XGBoost early stopping.")

        validation_size = max(1, int(len(features) * self.validation_fraction))
        validation_size = min(validation_size, len(features) - 2)
        split_at = len(features) - validation_size
        self.estimator.fit(
            features[:split_at] if isinstance(features, np.ndarray) else features.iloc[:split_at],
            target[:split_at] if isinstance(target, np.ndarray) else target.iloc[:split_at],
            eval_set=[
                (
                    features[split_at:] if isinstance(features, np.ndarray) else features.iloc[split_at:],
                    target[split_at:] if isinstance(target, np.ndarray) else target.iloc[split_at:],
                )
            ],
            verbose=False,
        )
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.estimator.predict(features)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator.feature_importances_
