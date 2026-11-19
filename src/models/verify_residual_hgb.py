from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.residual_hgb import EXPECTED_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready" / "X_train.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Artifact not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(X_TRAIN_PATH).head(5)
    if X.columns.tolist() != EXPECTED_FEATURES:
        raise ValueError(
            f"Input schema mismatch: expected {EXPECTED_FEATURES}, "
            f"got {X.columns.tolist()}."
        )

    predictions = np.asarray(model.predict(X), dtype=float)
    if len(predictions) != len(X):
        raise ValueError("Prediction row count does not match input row count.")
    if not np.isfinite(predictions).all():
        raise ValueError("Artifact generated non-finite predictions.")

    print(f"Loaded artifact: {MODEL_PATH}")
    print(f"Prediction count: {len(predictions)}")
    print(predictions)


if __name__ == "__main__":
    main()
