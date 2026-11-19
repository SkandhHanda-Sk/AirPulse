from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.residual_hgb import ANCHOR_COLUMN, EXPECTED_FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model_ready"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_evaluation" / "final_test"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "residual_hgb_model_info.json"
TEST_MARKER_PATH = OUTPUT_DIR / "test_evaluation_executed.json"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        raise ValueError("No valid test rows remain for metric calculation.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Test metric calculation received non-finite values.")
    return {
        "samples": len(y_true),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time final held-out test evaluation."
    )
    parser.add_argument(
        "--confirm-held-out-test",
        action="store_true",
        help="Explicitly confirm that validation is complete and this is the first test run.",
    )
    args = parser.parse_args()
    if not args.confirm_held_out_test:
        raise RuntimeError(
            "Held-out test evaluation requires --confirm-held-out-test."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_MARKER_PATH.exists():
        raise RuntimeError(
            "Final test evaluation has already been executed. "
            "Any code or model change invalidates the untouched-test policy."
        )
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(f"Model freeze information not found: {MODEL_INFO_PATH}")
    model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    if model_info.get("status") != "frozen_after_validation":
        raise RuntimeError(
            "Residual HGB model is not frozen after validation. "
            "Run validation evaluation before test evaluation."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Residual HGB artifact not found: {MODEL_PATH}")

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test_df = pd.read_csv(DATA_DIR / "y_test.csv")
    metadata_test = pd.read_csv(DATA_DIR / "metadata_test.csv")
    if list(X_test.columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Test feature schema mismatch.\n"
            f"Expected={EXPECTED_FEATURES}\nActual={X_test.columns.tolist()}."
        )
    if list(y_test_df.columns) != [TARGET_COLUMN]:
        raise ValueError(
            f"Test target schema mismatch: {y_test_df.columns.tolist()}."
        )
    if not (len(X_test) == len(y_test_df) == len(metadata_test)):
        raise ValueError(
            "Test row mismatch: "
            f"X_test={len(X_test)}, y_test={len(y_test_df)}, "
            f"metadata_test={len(metadata_test)}."
        )

    y_test = pd.to_numeric(
        y_test_df[TARGET_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)
    persistence = pd.to_numeric(
        X_test[ANCHOR_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)
    numeric_values = X_test.select_dtypes(include=[np.number]).to_numpy()
    valid_mask = (
        np.isfinite(y_test)
        & np.isfinite(persistence)
        & np.isfinite(numeric_values).all(axis=1)
    )

    excluded = int((~valid_mask).sum())
    X_match = X_test.loc[valid_mask].reset_index(drop=True)
    y_match = y_test[valid_mask]
    metadata_match = metadata_test.loc[valid_mask].reset_index(drop=True)
    persistence_match = persistence[valid_mask]

    residual_model = joblib.load(MODEL_PATH)
    residual_prediction = np.asarray(
        residual_model.predict(X_match),
        dtype=float,
    )
    if not np.isfinite(residual_prediction).all():
        raise ValueError("Residual HGB generated non-finite test predictions.")

    predictions = pd.DataFrame({
        "station_id": metadata_match["station_id"].astype(str),
        "period_start": metadata_match["period_start"],
        "target_timestamp": metadata_match["target_timestamp"],
        "actual_pm25": y_match,
        "persistence_prediction": persistence_match,
        "residual_hgb_prediction": residual_prediction,
    })
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
    overall_rows = [
        {"model": model_name, **metrics(y_match, y_pred)}
        for model_name, y_pred in model_predictions.items()
    ]
    overall = pd.DataFrame(overall_rows)

    station_rows = []
    for station_id in sorted(predictions["station_id"].unique()):
        station = predictions[predictions["station_id"] == station_id]
        actual = station["actual_pm25"].to_numpy(dtype=float)
        for model_name, column in [
            ("Persistence", "persistence_prediction"),
            ("Residual_HistGradientBoosting_L1", "residual_hgb_prediction"),
        ]:
            station_rows.append({
                "station_id": station_id,
                "model": model_name,
                **metrics(actual, station[column].to_numpy(dtype=float)),
            })
    station = pd.DataFrame(station_rows)

    persistence_row = overall[
        overall["model"] == "Persistence"
    ].iloc[0]
    residual_row = overall[
        overall["model"] == "Residual_HistGradientBoosting_L1"
    ].iloc[0]
    improvement = pd.DataFrame([{
        "metric": "MAE",
        "persistence": persistence_row["MAE"],
        "residual_hgb": residual_row["MAE"],
        "absolute_change": residual_row["MAE"] - persistence_row["MAE"],
        "percent_change": 100
        * (residual_row["MAE"] - persistence_row["MAE"])
        / persistence_row["MAE"],
    }, {
        "metric": "RMSE",
        "persistence": persistence_row["RMSE"],
        "residual_hgb": residual_row["RMSE"],
        "absolute_change": residual_row["RMSE"] - persistence_row["RMSE"],
        "percent_change": 100
        * (residual_row["RMSE"] - persistence_row["RMSE"])
        / persistence_row["RMSE"],
    }])

    residual_errors = predictions["residual_hgb_error"]
    residual_summary = pd.DataFrame([{
        "model": "Residual_HistGradientBoosting_L1",
        "error_mean": residual_errors.mean(),
        "error_std": residual_errors.std(),
        "absolute_error_mean": residual_errors.abs().mean(),
        "absolute_error_p95": residual_errors.abs().quantile(0.95),
    }])

    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "test_model_comparison.csv", index=False)
    station.to_csv(OUTPUT_DIR / "test_station_comparison.csv", index=False)
    improvement.to_csv(OUTPUT_DIR / "test_improvement.csv", index=False)
    residual_summary.to_csv(OUTPUT_DIR / "test_residual_summary.csv", index=False)

    summary = {
        "evaluation": "final_held_out_test",
        "model": "Persistence-Anchored Residual HistGradientBoosting",
        "rows_requested": len(X_test),
        "rows_evaluated": len(y_match),
        "rows_excluded": excluded,
        "exclusion_reason": "Non-finite target, anchor, or numeric feature values",
        "overall_metrics": overall.to_dict(orient="records"),
        "improvement_over_persistence": improvement.to_dict(orient="records"),
    }
    TEST_MARKER_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    model_info["test"] = summary
    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nStation-wise test comparison:")
    print(station.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nExcluded test rows: {excluded}")
    print(f"Final held-out test results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
