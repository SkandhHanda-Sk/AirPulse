# Training logic for persistence-anchored residual HGB

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.models.residual_hgb import (
    ANCHOR_COLUMN,
    EXPECTED_FEATURES,
    HGB_PARAMS,
    TARGET_COLUMN,
    build_residual_hgb_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_model_info.json"


def load_training_data():
    required = [
        DATA_DIR / "X_train.csv",
        DATA_DIR / "y_train.csv",
        DATA_DIR / "metadata_train.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing training files: {', '.join(missing)}"
        )

    X = pd.read_csv(required[0])
    y_df = pd.read_csv(required[1])
    metadata = pd.read_csv(required[2])

    if X.empty or y_df.empty or metadata.empty:
        raise ValueError("One or more training files are empty.")
    if list(X.columns) != EXPECTED_FEATURES:
        raise ValueError(
            "Training feature columns do not match the 18-feature schema:\n"
            f"Expected={EXPECTED_FEATURES}\nActual={X.columns.tolist()}"
        )
    if list(y_df.columns) != [TARGET_COLUMN]:
        raise ValueError(
            f"Training target must contain only '{TARGET_COLUMN}'. "
            f"Actual columns={y_df.columns.tolist()}."
        )
    if not (len(X) == len(y_df) == len(metadata)):
        raise ValueError(
            f"Training row mismatch: X={len(X)}, y={len(y_df)}, "
            f"metadata={len(metadata)}."
        )

    y = pd.to_numeric(y_df[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    for column in EXPECTED_FEATURES:
        if column == "station_id":
            continue
        pd.to_numeric(X[column], errors="raise")

    if X.isna().any().any():
        raise ValueError("Missing values found in training features.")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite target values found in training data.")
    if not np.isfinite(X[ANCHOR_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite pm25_at_origin values found in training data.")
    if set(X["station_id"].dropna()) != {"site_113", "site_115", "site_117"}:
        raise ValueError(
            "Training station IDs must contain site_113, site_115, and site_117."
        )
    if "station_id" not in metadata or len(metadata) != len(X):
        raise ValueError("Training metadata station_id alignment is invalid.")

    return X.reset_index(drop=True), y, metadata.reset_index(drop=True)


def main() -> None:
    X_train, y_train, metadata_train = load_training_data()
    model = build_residual_hgb_pipeline()

    print("Training persistence-anchored residual HistGradientBoosting...")
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_info = {
        "model_name": "Persistence-Anchored Residual HistGradientBoosting",
        "model_type": "PersistenceAnchorRegressor",
        "status": "trained_not_validated",
        "artifact_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "target": TARGET_COLUMN,
        "anchor": ANCHOR_COLUMN,
        "feature_list": EXPECTED_FEATURES,
        "feature_count": len(EXPECTED_FEATURES),
        "training": {
            "rows": len(X_train),
            "stations": sorted(metadata_train["station_id"].unique().tolist()),
            "X_file": "data/processed/model_ready/X_train.csv",
            "y_file": "data/processed/model_ready/y_train.csv",
            "metadata_file": "data/processed/model_ready/metadata_train.csv",
        },
        "configuration": HGB_PARAMS,
        "preprocessing": {
            "numeric": "passthrough",
            "categorical": "most-frequent imputation followed by OneHotEncoder",
            "handle_unknown": "ignore",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    print(f"Saved artifact: {MODEL_PATH}")
    print(f"Saved model information: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()
