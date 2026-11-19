from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.models.train_residual_hgb import (
    EXPECTED_FEATURES,
    HGB_PARAMS,
    TARGET_COLUMN,
    build_residual_hgb_pipeline,
    load_training_data,
)

# MLFlow tracking - file-based (no separate server needed)
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AirPulse_PM25_Forecasting")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_model_info.json"


def main() -> None:
    X_train, y_train, metadata_train = load_training_data()

    model = build_residual_hgb_pipeline()
    print("Training persistence-anchored residual HistGradientBoosting...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    mae = float(mean_absolute_error(y_train, train_pred))
    rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    r2 = float(r2_score(y_train, train_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    run_id = None
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(HGB_PARAMS)
            mlflow.log_param("feature_count", len(EXPECTED_FEATURES))
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param(
                "stations",
                ",".join(sorted(metadata_train["station_id"].unique().tolist())),
            )
            mlflow.log_metrics({"train_mae": mae, "train_rmse": rmse, "train_r2": r2})
            mlflow.sklearn.log_model(
                sk_model=model,
                name="residual-hgb-model",
                registered_model_name="airpulse-pm25-forecast",
            )
            try:
                client = MlflowClient()
                registered_versions = client.search_model_versions(
                    "name='airpulse-pm25-forecast'"
                )
                current_model_version = next(
                    (v for v in registered_versions if v.run_id == run_id), None
                )
                if current_model_version:
                    client.set_registered_model_alias(
                        name="airpulse-pm25-forecast",
                        alias="champion",
                        version=current_model_version.version,
                    )
                    print(
                        f"Registered model: airpulse-pm25-forecast "
                        f"version {current_model_version.version} as champion"
                    )
            except Exception as ex:
                print(f"MLflow model registration warning: {ex}")
            print(f"MLflow run logged: {run_id}")
    except Exception as ex:
        print(f"MLflow logging warning (non-fatal): {ex}")

    model_info = {
        "model_name": "Persistence-Anchored Residual HistGradientBoosting",
        "model_type": "PersistenceAnchorRegressor",
        "status": "trained",
        "artifact_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "target": TARGET_COLUMN,
        "feature_list": EXPECTED_FEATURES,
        "feature_count": len(EXPECTED_FEATURES),
        "training": {
            "rows": len(X_train),
            "stations": sorted(metadata_train["station_id"].unique().tolist()),
        },
        "configuration": HGB_PARAMS,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "mlflow_run_id": run_id,
        "mlflow_experiment": "AirPulse_PM25_Forecasting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    print(f"Model info saved to {MODEL_INFO_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
