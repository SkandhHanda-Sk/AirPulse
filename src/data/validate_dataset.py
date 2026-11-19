
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLUMNS = [
    "station_id",
    "parameter_name",
    "unit",
    "period_start",
    "mean",
    "min",
    "max",
    "n",
]


def load_dataset():
    files = sorted(RAW_DIR.glob("*.csv"))

    if len(files) != 12:
        raise ValueError(f"Expected 12 CSV files, found {len(files)}")

    dataframes = []

    for file in files:
        df = pd.read_csv(file)

        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {file.name}")

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def validate_dataset(df):
    print("\n========== DATASET INTEGRITY REPORT ==========")

    print("\n1. Dataset Overview")
    print("Total records:", len(df))
    print("Total columns:", len(df.columns))

    print("\n2. Records Per Station")
    print(df["station_id"].value_counts().sort_index())

    print("\n3. Missing Values")
    print(df.isnull().sum())

    print("\n4. Data Types")
    print(df.dtypes)

    print("\n5. Duplicate Station-Hour Records")
    duplicates = df.duplicated(
        subset=["station_id", "period_start"]
    ).sum()
    print("Duplicate records:", duplicates)

    print("\n6. Timestamp Validation")
    timestamps = pd.to_datetime(
        df["period_start"], errors="coerce"
    )

    print("Invalid timestamps:", timestamps.isna().sum())
    print("Earliest timestamp:", timestamps.min())
    print("Latest timestamp:", timestamps.max())
    print("Records outside 2024:", (~timestamps.dt.year.eq(2024)).sum())

    print("\n7. PM2.5 Value Validation")
    numeric_pm25 = pd.to_numeric(df["mean"], errors="coerce")

    print("Non-numeric PM2.5 values:", numeric_pm25.isna().sum())
    print("Negative PM2.5 values:", (numeric_pm25 < 0).sum())
    print("Minimum PM2.5:", numeric_pm25.min())
    print("Maximum PM2.5:", numeric_pm25.max())

    print("\n8. Unexpected Values")
    print("Unexpected stations:",
          sorted(set(df["station_id"]) - {"site_113", "site_115", "site_117"}))
    print("Unexpected parameters:",
          sorted(set(df["parameter_name"]) - {"PM2.5"}))
    print("Unexpected units:",
          sorted(set(df["unit"]) - {"µg/m³"}))

    print("\n9. Hourly Timestamp Alignment")
    valid_timestamps = timestamps.dropna()
    misaligned = (
        (valid_timestamps.dt.minute != 0)
        | (valid_timestamps.dt.second != 0)
    ).sum()
    print("Non-hourly-aligned timestamps:", misaligned)

    print("\n========== VALIDATION CHECKS COMPLETED ==========")


def check_missing_hours(df):
    print("\n10. Missing Hourly Observations")

    expected_hours = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2024-12-31 23:00:00",
        freq="h"
    )

    total_missing = 0

    for station in sorted(df["station_id"].unique()):
        station_data = df[df["station_id"] == station]

        actual_hours = pd.DatetimeIndex(
            pd.to_datetime(station_data["period_start"])
        )

        missing_hours = expected_hours.difference(actual_hours)
        missing_count = len(missing_hours)

        total_missing += missing_count

        print(f"\nStation: {station}")
        print("Available records:", len(actual_hours))
        print("Missing hours:", missing_count)
        print("First five missing timestamps:")
        print(list(missing_hours[:5]))

    print("\nTotal missing station-hours:", total_missing)
    print(
        "Expected total records:",
        len(expected_hours) * df["station_id"].nunique()
    )


def inspect_extreme_values(df):
    print("\n11. Extreme PM2.5 Readings")

    columns = [
        "station_id",
        "period_start",
        "mean"
    ]

    print("\nTop 15 highest PM2.5 readings:")

    extreme = (
        df.nlargest(15, "mean")[columns]
        .sort_values("mean", ascending=False)
    )

    print(extreme.to_string(index=False))

    print("\nPM2.5 Descriptive Statistics:")
    print(df["mean"].describe())

    print("\nReadings above selected thresholds:")

    for threshold in [100, 200, 300, 500]:
        count = (df["mean"] > threshold).sum()
        print(f"Above {threshold} µg/m³: {count}")


if __name__ == "__main__":
    dataset = load_dataset()
    validate_dataset(dataset)
    check_missing_hours(dataset)
    inspect_extreme_values(dataset)