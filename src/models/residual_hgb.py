# Persistence-anchored residual HGB model implementation

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "target_pm25_next_hour"
ANCHOR_COLUMN = "pm25_at_origin"
CATEGORICAL_FEATURES = ["station_id"]
NUMERIC_FEATURES = [
    "pm25_at_origin",
    "pm25_1h_before",
    "pm25_2h_before",
    "pm25_23h_before",
    "pm25_rolling_mean_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_observation_count_24h",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]
EXPECTED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Loss parameter name compatibility across scikit-learn versions
if hasattr(HistGradientBoostingRegressor, "_VALID_LOSSES") and "absolute_error" not in HistGradientBoostingRegressor._VALID_LOSSES:
    DEFAULT_LOSS = "least_absolute_deviation"
else:
    DEFAULT_LOSS = "absolute_error"

HGB_PARAMS = {
    "loss": DEFAULT_LOSS,
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


class PersistenceAnchorRegressor(BaseEstimator, RegressorMixin):
    """Predict actual PM2.5 by adding a residual correction to pm25_at_origin."""

    def __init__(
        self,
        base_pipeline: Any = None,
        anchor_col: str = ANCHOR_COLUMN,
    ):
        self.base_pipeline = base_pipeline
        self.anchor_col = anchor_col

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if self.anchor_col not in X.columns:
            raise ValueError(
                "Anchor column '{}' is missing from model inputs.".format(self.anchor_col)
            )

        anchor = pd.to_numeric(
            X[self.anchor_col],
            errors="raise",
        ).to_numpy(dtype=float)
        actual = np.asarray(y, dtype=float)
        if len(X) != len(actual):
            raise ValueError(
                "Feature/target row mismatch: X={}, y={}.".format(len(X), len(actual))
            )
        if not np.isfinite(anchor).all():
            raise ValueError("Non-finite anchor values found in training data.")
        if not np.isfinite(actual).all():
            raise ValueError("Non-finite target values found in training data.")

        residual_target = actual - anchor
        self.estimator_ = clone(self.base_pipeline)
        self.estimator_.fit(X, residual_target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if self.anchor_col not in X.columns:
            raise ValueError(
                "Anchor column '{}' is missing from model inputs.".format(self.anchor_col)
            )

        anchor = pd.to_numeric(
            X[self.anchor_col],
            errors="raise",
        ).to_numpy(dtype=float)
        if not np.isfinite(anchor).all():
            raise ValueError("Non-finite anchor values found in prediction data.")

        residual_prediction = np.asarray(
            self.estimator_.predict(X),
            dtype=float,
        )
        prediction = anchor + residual_prediction
        if not np.isfinite(prediction).all():
            raise ValueError("Estimator generated non-finite predictions.")
        return prediction


def build_residual_hgb_pipeline():
    """Build the preprocessing plus residual HGB pipeline."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    hgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(**HGB_PARAMS)),
        ]
    )

    return PersistenceAnchorRegressor(
        base_pipeline=hgb_pipeline,
        anchor_col=ANCHOR_COLUMN,
    )
