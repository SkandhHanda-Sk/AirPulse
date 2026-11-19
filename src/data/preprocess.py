
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "hourly_panel.csv"

TIMEZONE = "Asia/Kolkata"

START_DATE = "2024-01-01 00:00"
END_DATE = "2024-12-31 23:00"


# --------------------------------------------------
# 2. LOAD RAW DATA
# --------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if len(csv_files) != 12:
        raise ValueError(
            f"Expected 12 monthly CSV files, "
            f"found {len(csv_files)}."
        )

    frames = [pd.read_csv(file) for file in csv_files]

    df = pd.concat(frames, ignore_index=True)

    required_columns = {
        "station_id",
        "parameter_name",
        "unit",
        "period_start",
        "mean",
        "min",
        "max",
        "n",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="raise"
    )

    df["period_start"] = (
        df["period_start"].dt.tz_localize(TIMEZONE)
    )

    return df


# --------------------------------------------------
# 3. VALIDATE SOURCE DATA
# --------------------------------------------------

def validate_raw_data(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError("Raw dataset is empty.")

    if df["mean"].isna().any():
        raise ValueError(
            "Unexpected null PM2.5 values in raw data."
        )

    if (df["mean"] < 0).any():
        raise ValueError(
            "Negative PM2.5 observations detected."
        )

    if df.duplicated(
        subset=["station_id", "period_start"]
    ).any():
        raise ValueError(
            "Duplicate station-hour records detected."
        )

    if not df["parameter_name"].eq("PM2.5").all():
        raise ValueError("Unexpected parameter found.")

    if not df["unit"].eq("µg/m³").all():
        raise ValueError("Unexpected measurement unit found.")

    if df["period_start"].dt.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")


# --------------------------------------------------
# 4. BUILD COMPLETE HOURLY PANEL
# --------------------------------------------------

def build_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Mark actual source observations before reindexing.
    df["is_observed"] = True

    station_ids = sorted(df["station_id"].unique())

    expected_timeline = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="h",
        tz=TIMEZONE,
    )

    full_index = pd.MultiIndex.from_product(
        [station_ids, expected_timeline],
        names=["station_id", "period_start"],
    )

    panel = (
        df.set_index(["station_id", "period_start"])
        .sort_index()
        .reindex(full_index)
        .reset_index()
    )

    # Rows introduced by reindexing are missing observations.
    panel["is_observed"] = (
        panel["is_observed"]
        .fillna(False)
        .astype(bool)
    )

    panel = panel.sort_values(
        ["station_id", "period_start"]
    ).reset_index(drop=True)

    return panel


# --------------------------------------------------
# 5. QUALITY CHECKS
# --------------------------------------------------

def validate_processed_data(
    panel: pd.DataFrame,
    original: pd.DataFrame,
) -> None:

    expected_hours = 8784
    expected_stations = original["station_id"].nunique()

    expected_rows = expected_hours * expected_stations

    if len(panel) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(panel)}."
        )

    if panel.duplicated(
        subset=["station_id", "period_start"]
    ).any():
        raise ValueError(
            "Duplicate station-hour rows in processed data."
        )

    observed_count = int(panel["is_observed"].sum())

    if observed_count != len(original):
        raise ValueError(
            "Observed record count does not match raw data."
        )

    missing_count = int((~panel["is_observed"]).sum())

    if missing_count + observed_count != len(panel):
        raise ValueError("Observation count mismatch.")

    if panel.loc[
        ~panel["is_observed"], "mean"
    ].notna().any():
        raise ValueError(
            "Inserted missing rows must have NaN PM2.5."
        )

    if panel.loc[
        panel["is_observed"], "mean"
    ].isna().any():
        raise ValueError(
            "An observed record has missing PM2.5."
        )

    # Verify every original measurement is preserved.
    original_values = (
        original.set_index(["station_id", "period_start"])["mean"]
        .sort_index()
    )

    processed_values = (
        panel.set_index(["station_id", "period_start"])["mean"]
        .reindex(original_values.index)
    )

    if not original_values.equals(processed_values):
        raise ValueError(
            "Original PM2.5 values changed during preprocessing."
        )


# --------------------------------------------------
# 6. SAVE REPORT AND PROCESSED DATA
# --------------------------------------------------

def main() -> None:

    print("Loading raw dataset...")
    raw_df = load_raw_data()

    print("Validating raw dataset...")
    validate_raw_data(raw_df)

    print("Building complete hourly timeline...")
    panel = build_hourly_panel(raw_df)

    print("Validating processed dataset...")
    validate_processed_data(panel, raw_df)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    panel.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPREPROCESSING COMPLETED")
    print("-" * 45)

    print(f"Raw records: {len(raw_df)}")
    print(f"Processed rows: {len(panel)}")

    print(
        "Observed rows:",
        int(panel["is_observed"].sum())
    )

    print(
        "Missing rows:",
        int((~panel["is_observed"]).sum())
    )

    print("\nStation-wise summary:")

    summary = (
        panel.groupby("station_id")
        .agg(
            total_hours=("period_start", "size"),
            observed_hours=("is_observed", "sum"),
            missing_hours=(
                "is_observed",
                lambda x: (~x).sum()
            ),
        )
    )

    print(summary)

    print(f"\nSaved processed dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
