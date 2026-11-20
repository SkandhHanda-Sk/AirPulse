from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.mlflow_config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.models.residual_hgb import ANCHOR_COLUMN, EXPECTED_FEATURES, TARGET_COLUMN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_evaluation"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_model_info.json"


def load_and_validate(split: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    X_path = DATA_DIR / f"X_{split}.csv"
    y_path = DATA_DIR / f"y_{split}.csv"
    metadata_path = DATA_DIR / f"metadata_{split}.csv"
    for path in (X_path, y_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required {split} file not found: {path}")

    X = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    metadata = pd.read_csv(metadata_path)
    if X.empty or y_df.empty or metadata.empty:
        raise ValueError(f"{split} evaluation files contain empty data.")
    if list(X.columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"{split} feature schema mismatch.\n"
            f"Expected={EXPECTED_FEATURES}\nActual={X.columns.tolist()}"
        )
    if list(y_df.columns) != [TARGET_COLUMN]:
        raise ValueError(
            f"{split} target schema mismatch: {y_df.columns.tolist()}"
        )
    if not (len(X) == len(y_df) == len(metadata)):
        raise ValueError(
            f"{split} row mismatch: X={len(X)}, y={len(y_df)}, "
            f"metadata={len(metadata)}"
        )

    y = pd.to_numeric(y_df[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    return X.reset_index(drop=True), y, metadata.reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if len(y_true) != len(y_pred):
        raise ValueError("Metric calculation received mismatched row counts.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric calculation received non-finite values.")
    return {
        "samples": len(y_true),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def validation_quality_gate_passed(
    residual_mae: float,
    persistence_mae: float,
) -> bool:
    """Require the model to improve on the existing persistence baseline."""
    return residual_mae < persistence_mae


def main(residual_model: Any | None = None) -> bool:
    if residual_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Residual HGB artifact not found: {MODEL_PATH}. "
                "Run python train.py first."
            )
        residual_model = joblib.load(MODEL_PATH)
    X_validation, y_validation, metadata = load_and_validate("validation")

    persistence = pd.to_numeric(
        X_validation[ANCHOR_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)
    finite_mask = (
        np.isfinite(y_validation)
        & np.isfinite(persistence)
        & np.isfinite(X_validation.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)
    )
    excluded = int((~finite_mask).sum())
    X_match = X_validation.loc[finite_mask].reset_index(drop=True)
    y_match = y_validation[finite_mask]
    metadata_match = metadata.loc[finite_mask].reset_index(drop=True)
    persistence_match = persistence[finite_mask]

    residual_prediction = np.asarray(
        residual_model.predict(X_match),
        dtype=float,
    )
    if not np.isfinite(residual_prediction).all():
        raise ValueError("Residual HGB generated non-finite validation predictions.")

    predictions = pd.DataFrame(
        {
            "station_id": metadata_match["station_id"].astype(str),
            "period_start": metadata_match["period_start"],
            "target_timestamp": metadata_match["target_timestamp"],
            "actual_pm25": y_match,
            "persistence_prediction": persistence_match,
            "residual_hgb_prediction": residual_prediction,
        }
    )
    predictions["persistence_residual"] = (
        predictions["actual_pm25"] - predictions["persistence_prediction"]
    )
    predictions["residual_hgb_error"] = (
        predictions["actual_pm25"] - predictions["residual_hgb_prediction"]
    )
    model_predictions = {
        "Persistence": persistence_match,
        "Residual_HistGradientBoosting_L1": residual_prediction,
    }
    overall_rows = []
    for model_name, y_pred in model_predictions.items():
        overall_rows.append(
            {"model": model_name, **metrics(y_match, y_pred)}
        )
    overall = pd.DataFrame(overall_rows)

    # Log validation metrics to MLflow tracking server
    residual_row = overall[overall["model"] == "Residual_HistGradientBoosting_L1"].iloc[0]
    persistence_row = overall[overall["model"] == "Persistence"].iloc[0]
    mlflow.log_metric("val_mae", float(residual_row["MAE"]))
    mlflow.log_metric("val_rmse", float(residual_row["RMSE"]))
    mlflow.log_metric("val_r2", float(residual_row["R2"]))
    mlflow.log_metric("val_samples", int(residual_row["samples"]))
    mlflow.log_param("validation_model", "Residual_HistGradientBoosting_L1")
    mlflow.log_param("baseline_model", "Persistence")

    residual_mae = float(residual_row["MAE"])
    persistence_mae = float(persistence_row["MAE"])
    quality_gate_passed = validation_quality_gate_passed(
        residual_mae,
        persistence_mae,
    )
    mlflow.log_param("quality_gate_passed", str(quality_gate_passed))
    mlflow.log_param("quality_gate_metric", "val_mae")
    mlflow.log_param(
        "quality_gate_comparison",
        "val_mae < persistence_val_mae",
    )
    mlflow.log_metric("quality_gate_persistence_mae", persistence_mae)
    mlflow.log_metric("quality_gate_mae_delta", residual_mae - persistence_mae)

    if quality_gate_passed:
        print(
            f"QUALITY GATE PASSED: val_mae={residual_mae:.4f} "
            f"< persistence_mae={persistence_mae:.4f}."
        )
    else:
        print(
            f"QUALITY GATE FAILED: val_mae={residual_mae:.4f} "
            f">= persistence_mae={persistence_mae:.4f}."
        )

    station_rows = []
    for station_id in sorted(predictions["station_id"].unique()):
        station = predictions[predictions["station_id"] == station_id]
        actual = station["actual_pm25"].to_numpy(dtype=float)
        for model_name, column in [
            ("Persistence", "persistence_prediction"),
            ("Residual_HistGradientBoosting_L1", "residual_hgb_prediction"),
        ]:
            station_rows.append(
                {
                    "station_id": station_id,
                    "model": model_name,
                    **metrics(actual, station[column].to_numpy(dtype=float)),
                }
            )
    station = pd.DataFrame(station_rows)

    persistence_mae = overall.loc[
        overall["model"] == "Persistence", "MAE"
    ].iloc[0]
    residual_mae = overall.loc[
        overall["model"] == "Residual_HistGradientBoosting_L1", "MAE"
    ].iloc[0]
    persistence_rmse = overall.loc[
        overall["model"] == "Persistence", "RMSE"
    ].iloc[0]
    residual_rmse = overall.loc[
        overall["model"] == "Residual_HistGradientBoosting_L1", "RMSE"
    ].iloc[0]

    improvement = pd.DataFrame(
        [{
            "metric": "MAE",
            "persistence": persistence_mae,
            "residual_hgb": residual_mae,
            "absolute_change": residual_mae - persistence_mae,
            "percent_change": 100 * (residual_mae - persistence_mae) / persistence_mae,
        }, {
            "metric": "RMSE",
            "persistence": persistence_rmse,
            "residual_hgb": residual_rmse,
            "absolute_change": residual_rmse - persistence_rmse,
            "percent_change": 100 * (residual_rmse - persistence_rmse) / persistence_rmse,
        }]
    )

    residual_errors = predictions["residual_hgb_error"]
    residual_summary = pd.DataFrame([{
        "model": "Residual_HistGradientBoosting_L1",
        "error_mean": residual_errors.mean(),
        "error_std": residual_errors.std(),
        "absolute_error_mean": residual_errors.abs().mean(),
        "absolute_error_p95": residual_errors.abs().quantile(0.95),
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT_DIR / "residual_hgb_validation_predictions.csv",
        index=False,
    )
    overall.to_csv(
        OUTPUT_DIR / "residual_hgb_validation_model_comparison.csv",
        index=False,
    )
    station.to_csv(
        OUTPUT_DIR / "residual_hgb_validation_station_comparison.csv",
        index=False,
    )
    improvement.to_csv(
        OUTPUT_DIR / "residual_hgb_validation_improvement.csv",
        index=False,
    )
    residual_summary.to_csv(
        OUTPUT_DIR / "residual_hgb_validation_residual_summary.csv",
        index=False,
    )

    model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    model_info["status"] = "frozen_after_validation"
    model_info["validation"] = {
        "rows_requested": len(X_validation),
        "rows_evaluated": len(y_match),
        "rows_excluded": excluded,
        "exclusion_reason": "Non-finite target, anchor, or numeric feature values",
        "metrics": overall.to_dict(orient="records"),
        "station_metrics_file": str(
            (OUTPUT_DIR / "residual_hgb_validation_station_comparison.csv").relative_to(PROJECT_ROOT)
        ),
        "predictions_file": str(
            (OUTPUT_DIR / "residual_hgb_validation_predictions.csv").relative_to(PROJECT_ROOT)
        ),
    }
    model_info["quality_gate"] = {
        "metric": "MAE",
        "candidate_mae": residual_mae,
        "persistence_mae": persistence_mae,
        "passed": quality_gate_passed,
    }
    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nStation-wise validation comparison:")
    print(station.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nExcluded rows: {excluded}")
    print("Selected model frozen for held-out test evaluation.")
    return quality_gate_passed


if __name__ == "__main__":
    main()
