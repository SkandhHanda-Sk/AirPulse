
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


def evaluate_station(df, split_name, station_id):
    valid = df[
        (df["station_id"] == station_id)
        & df["pm25_at_origin"].notna()
        & df["target_pm25_next_hour"].notna()
    ]

    if valid.empty:
        return None

    actual = valid["target_pm25_next_hour"].to_numpy(dtype=float)
    predicted = valid["pm25_at_origin"].to_numpy(dtype=float)

    return {
        "split": split_name,
        "station_id": station_id,
        "evaluated_rows": len(valid),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    results = []

    for split_name, file_path in SPLIT_FILES.items():
        df = pd.read_csv(file_path)

        for station_id in sorted(df["station_id"].dropna().unique()):
            result = evaluate_station(df, split_name, station_id)

            if result is not None:
                results.append(result)

    results_df = pd.DataFrame(results)

    print("\nSTATION-WISE PERSISTENCE BASELINE")
    print("=" * 75)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_path = PROCESSED_DIR / "station_baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()
