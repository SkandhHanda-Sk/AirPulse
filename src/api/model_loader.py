"""Singleton model loader for the trained Residual HGB pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.residual_hgb import ANCHOR_COLUMN, EXPECTED_FEATURES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_CANDIDATES = (
    _PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib",
    _PROJECT_ROOT / "models" / "residual_hgb_pipeline.joblib",
)

_model: Any = None


def _resolve_model_path() -> Path:
    for path in _MODEL_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in _MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"Model artifact not found. Searched: {searched}. "
        "Run 'python -m src.models.train_residual_hgb' or 'python train.py' first."
    )


def load_model() -> Any:
    """Load and cache the trained model artifact."""
    global _model
    if _model is None:
        _model = joblib.load(_resolve_model_path())
    return _model


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