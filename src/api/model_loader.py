"""Singleton model loader that loads the @champion model from MLflow Model Registry."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd

from src.mlflow_config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from src.models.residual_hgb import ANCHOR_COLUMN, EXPECTED_FEATURES

_model: Any = None
_model_source: str = ""


def load_model() -> Any:
    """Load and cache the @champion model from MLflow Model Registry."""
    global _model, _model_source
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
        _model = mlflow.pyfunc.load_model(model_uri)
        _model_source = model_uri
    return _model


def get_model_source() -> str:
    """Return the URI of the currently loaded model."""
    return _model_source


def get_feature_columns() -> list[str]:
    """Return the expected feature column names in order."""
    return list(EXPECTED_FEATURES)


def get_anchor_column() -> str:
    """Return the anchor column name used by the persistence wrapper."""
    return ANCHOR_COLUMN


def predict(model: Any, features_df: pd.DataFrame) -> list[float]:
    """Run prediction on a validated feature DataFrame.

    Validates that the input DataFrame contains exactly the expected
    feature columns before calling the model.
    """
    missing = [c for c in EXPECTED_FEATURES if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    extra = [c for c in features_df.columns if c not in EXPECTED_FEATURES]
    if extra:
        raise ValueError(f"Unexpected feature columns: {extra}")

    predictions = model.predict(features_df[EXPECTED_FEATURES])
    return [float(p) for p in predictions]