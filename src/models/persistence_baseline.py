
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_FILES = {
    "TRAIN": PROCESSED_DIR / "train.csv",
    "VALIDATION": PROCESSED_DIR / "validation.csv",
    "TEST": PROCESSED_DIR / "test.csv",
}

CURRENT_FEATURE = "pm25_at_origin"
TARGET_COLUMN = "target_pm25_next_hour"


def load_split(file_path):
    """Load a dataset split and parse timestamps."""
    df = pd.read_csv(file_path)

    df["period_start"] = pd.to_datetime(
        df["period_start"], utc=True
    ).dt.tz_convert("Asia/Kolkata")

    df["target_timestamp"] = pd.to_datetime(
        df["target_timestamp"], utc=True
    ).dt.tz_convert("Asia/Kolkata")

    return df


def evaluate_persistence(df, split_name):
    """Evaluate the persistence baseline on valid observations."""

    required_columns = [
        CURRENT_FEATURE,
        TARGET_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name}: Missing columns {missing_columns}"
        )

    valid_mask = (
        df[CURRENT_FEATURE].notna()
        & df[TARGET_COLUMN].notna()
    )

    valid_df = df.loc[valid_mask].copy()

    if valid_df.empty:
        raise ValueError(
            f"{split_name}: No valid rows for evaluation."
        )

    y_true = valid_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = valid_df[CURRENT_FEATURE].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    results = {
        "split": split_name,
        "total_rows": len(df),
        "evaluated_rows": len(valid_df),
        "excluded_rows": len(df) - len(valid_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return results


def main():
    print("AIRPULSE - PERSISTENCE BASELINE")
    print("=" * 55)

    results = []

    for split_name, file_path in SPLIT_FILES.items():

        print(f"\nEvaluating {split_name}...")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        df = load_split(file_path)

        split_results = evaluate_persistence(
            df,
            split_name,
        )

        results.append(split_results)

        print(f"Total rows:     {split_results['total_rows']}")
        print(f"Evaluated rows: {split_results['evaluated_rows']}")
        print(f"Excluded rows:  {split_results['excluded_rows']}")
        print(f"MAE:            {split_results['MAE']:.4f}")
        print(f"RMSE:           {split_results['RMSE']:.4f}")
        print(f"R2:             {split_results['R2']:.4f}")

    results_df = pd.DataFrame(results)

    output_path = PROCESSED_DIR / "persistence_baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print("BASELINE EVALUATION COMPLETED")
    print("=" * 55)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()
