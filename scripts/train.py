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

from src.mlflow_config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from src.models.evaluate_residual_hgb import main as evaluate_validation
from src.models.train_residual_hgb import (
    EXPECTED_FEATURES,
    HGB_PARAMS,
    TARGET_COLUMN,
    build_residual_hgb_pipeline,
    load_training_data,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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
    with mlflow.start_run(run_name="residual_hgb_training") as run:
        run_id = run.info.run_id
        mlflow.log_params(HGB_PARAMS)
        mlflow.log_param("feature_count", len(EXPECTED_FEATURES))
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param(
            "stations",
            ",".join(sorted(metadata_train["station_id"].unique().tolist())),
        )
        mlflow.log_param("registered_model_name", REGISTERED_MODEL_NAME)
        mlflow.set_tags(
            {
                "model_type": "PersistenceAnchorRegressor",
                "target": TARGET_COLUMN,
            }
        )
        mlflow.log_metrics({"train_mae": mae, "train_rmse": rmse, "train_r2": r2})

        model_info = {
            "model_name": "Persistence-Anchored Residual HistGradientBoosting",
            "model_type": "PersistenceAnchorRegressor",
            "status": "trained_not_validated",
            "artifact_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "target": TARGET_COLUMN,
            "feature_list": EXPECTED_FEATURES,
            "feature_count": len(EXPECTED_FEATURES),
            "training": {
                "rows": len(X_train),
                "stations": sorted(
                    metadata_train["station_id"].unique().tolist()
                ),
            },
            "configuration": HGB_PARAMS,
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
            "mlflow_run_id": run_id,
            "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        MODEL_INFO_PATH.write_text(
            json.dumps(model_info, indent=2),
            encoding="utf-8",
        )

        quality_gate_passed = evaluate_validation(residual_model=model)
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            await_registration_for=300,
        )

        client = MlflowClient()
        registered_versions = [
            version
            for version in client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
            if version.run_id == run_id
        ]
        if not registered_versions:
            raise RuntimeError(
                f"MLflow did not create a registered version for run {run_id}."
            )
        registered_version = max(
            registered_versions,
            key=lambda version: int(version.version),
        )
        registered_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=registered_version.version,
        )
        if registered_version.status != "READY":
            raise RuntimeError(
                f"Registered model version {registered_version.version} "
                f"is not ready: {registered_version.status}."
            )

        if quality_gate_passed:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="champion",
                version=registered_version.version,
            )
            print(
                f"Registered model {REGISTERED_MODEL_NAME} version "
                f"{registered_version.version} as @champion."
            )
        else:
            print(
                f"Registered model {REGISTERED_MODEL_NAME} version "
                f"{registered_version.version}; quality gate failed, "
                "so the existing @champion alias was retained."
            )

        model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        model_info["mlflow_model_version"] = registered_version.version
        model_info["mlflow_model_uri"] = logged_model.model_uri
        model_info["champion_alias_updated"] = quality_gate_passed
        MODEL_INFO_PATH.write_text(
            json.dumps(model_info, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(MODEL_INFO_PATH), artifact_path="metadata")
        print(f"MLflow run logged: {run_id}")

    print(f"Model info saved to {MODEL_INFO_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()



