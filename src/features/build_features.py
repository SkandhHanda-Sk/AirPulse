
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_panel.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "feature_dataset.csv"

TIMEZONE = "Asia/Kolkata"


# --------------------------------------------------
# 2. LOAD PROCESSED HOURLY PANEL
# --------------------------------------------------

def load_hourly_panel() -> pd.DataFrame:

    df = pd.read_csv(INPUT_FILE)

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="raise"
    )

    if df["period_start"].dt.tz is None:
        raise ValueError("Timezone information is missing.")

    if str(df["period_start"].dt.tz) not in [
        "pytz.FixedOffset(330)",
        "Asia/Kolkata",
        "UTC+05:30",
    ]:
        raise ValueError("Unexpected timestamp timezone.")

    df["is_observed"] = df["is_observed"].astype(bool)

    df = df.sort_values(
        ["station_id", "period_start"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# 3. VALIDATE HOURLY TIMELINE
# --------------------------------------------------

def validate_hourly_panel(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if df.duplicated(
        ["station_id", "period_start"]
    ).any():
        raise ValueError("Duplicate station-hour records found.")

    for station_id, group in df.groupby("station_id"):

        timestamps = group["period_start"]

        differences = timestamps.diff().dropna()

        if not (
            differences == pd.Timedelta(hours=1)
        ).all():
            raise ValueError(
                f"Non-hourly timeline found for {station_id}."
            )

    if len(df) != 26352:
        raise ValueError(
            f"Expected 26352 rows, found {len(df)}."
        )


# --------------------------------------------------
# 4. GENERATE FEATURES FOR ONE STATION
# --------------------------------------------------

def build_station_features(
    station_df: pd.DataFrame
) -> pd.DataFrame:

    station_df = station_df.copy()

    station_df = station_df.sort_values(
        "period_start"
    ).reset_index(drop=True)

    pm25 = station_df["mean"]

    # ----------------------------------------------
    # A. LAG FEATURES
    # ----------------------------------------------

    station_df["pm25_at_origin"] = pm25

    station_df["pm25_1h_before"] = pm25.shift(1)

    station_df["pm25_2h_before"] = pm25.shift(2)

    station_df["pm25_23h_before"] = pm25.shift(23)

    # ----------------------------------------------
    # B. PAST-ONLY ROLLING FEATURES
    # ----------------------------------------------

    
    # ----------------------------------------------
    # B. PAST-ONLY ROLLING FEATURES
    # ----------------------------------------------

    station_df["pm25_rolling_mean_3h"] = (
        pm25.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    station_df["pm25_rolling_mean_6h"] = (
        pm25.rolling(
            window=6,
            min_periods=6
        ).mean()
    )

    # Number of actual observations in the trailing 24-hour window.
    # Missing monitoring hours are not counted.
    station_df["pm25_rolling_observation_count_24h"] = (
        station_df["is_observed"]
        .astype(int)
        .rolling(
            window=24,
            min_periods=1
        )
        .sum()
        .astype(int)
    )

    # Calculate the mean when at least 18 of 24 readings exist.
    station_df["pm25_rolling_mean_24h"] = (
        pm25.rolling(
            window=24,
            min_periods=18
        ).mean()
    )

    # ----------------------------------------------
    # C. CALENDAR FEATURES
    # ----------------------------------------------

    station_df["hour"] = (
        station_df["period_start"].dt.hour
    )

    station_df["day_of_week"] = (
        station_df["period_start"].dt.day_of_week
    )

    station_df["month"] = (
        station_df["period_start"].dt.month
    )

    # ----------------------------------------------
    # D. CYCLICAL ENCODINGS
    # ----------------------------------------------

    station_df["hour_sin"] = np.sin(
        2 * np.pi * station_df["hour"] / 24
    )

    station_df["hour_cos"] = np.cos(
        2 * np.pi * station_df["hour"] / 24
    )

    station_df["day_of_week_sin"] = np.sin(
        2 * np.pi * station_df["day_of_week"] / 7
    )

    station_df["day_of_week_cos"] = np.cos(
        2 * np.pi * station_df["day_of_week"] / 7
    )

    station_df["month_sin"] = np.sin(
        2 * np.pi * (station_df["month"] - 1) / 12
    )

    station_df["month_cos"] = np.cos(
        2 * np.pi * (station_df["month"] - 1) / 12
    )

    # ----------------------------------------------
    # E. EXACT NEXT-HOUR TARGET
    # ----------------------------------------------

    station_df["target_timestamp"] = (
        station_df["period_start"]
        + pd.Timedelta(hours=1)
    )

    station_df["target_pm25_next_hour"] = (
        pm25.shift(-1)
    )

    station_df["target_is_observed"] = (
        station_df["is_observed"].shift(-1)
        .fillna(False)
        .astype(bool)
    )

    # ----------------------------------------------
    # F. SELECT FINAL COLUMNS
    # ----------------------------------------------

    feature_columns = [
        "station_id",
        "period_start",
        "target_timestamp",
        "is_observed",
        "target_is_observed",

        "pm25_at_origin",
        "pm25_1h_before",
        "pm25_2h_before",
        "pm25_23h_before",

        "pm25_rolling_mean_3h",
        "pm25_rolling_mean_6h",
        "pm25_rolling_observation_count_24h",
        "pm25_rolling_mean_24h",

        "hour",
        "day_of_week",
        "month",

        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",

        "target_pm25_next_hour",
    ]

    return station_df[feature_columns]


# --------------------------------------------------
# 5. BUILD COMPLETE FEATURE DATASET
# --------------------------------------------------

def build_feature_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:

    station_features = []

    for station_id, group in df.groupby("station_id"):

        print(f"Generating features: {station_id}")

        station_feature_df = build_station_features(group)

        station_features.append(station_feature_df)

    feature_df = pd.concat(
        station_features,
        ignore_index=True
    )

    feature_df = feature_df.sort_values(
        ["station_id", "period_start"]
    ).reset_index(drop=True)

    return feature_df


# --------------------------------------------------
# 6. FEATURE QUALITY CHECKS
# --------------------------------------------------

def validate_feature_dataset(
    original_df: pd.DataFrame,
    feature_df: pd.DataFrame
) -> None:

    if len(feature_df) != len(original_df):
        raise ValueError("Feature dataset row count mismatch.")

    if feature_df.duplicated(
        ["station_id", "period_start"]
    ).any():
        raise ValueError("Duplicate feature timestamps found.")

    # Every target must be exactly one hour after origin.
    expected_target_time = (
        feature_df["period_start"]
        + pd.Timedelta(hours=1)
    )

    if not (
        feature_df["target_timestamp"]
        == expected_target_time
    ).all():
        raise ValueError("Target timestamp alignment error.")

    # Check exact next-hour target values.
    
    # Validate 24-hour rolling observation counts.
    for station_id, group in feature_df.groupby("station_id"):

        group = group.sort_values("period_start")

        expected_counts = (
            original_df[
                original_df["station_id"] == station_id
            ]
            .sort_values("period_start")["is_observed"]
            .astype(int)
            .rolling(window=24, min_periods=1)
            .sum()
            .astype(int)
            .reset_index(drop=True)
        )

        actual_counts = (
            group["pm25_rolling_observation_count_24h"]
            .reset_index(drop=True)
        )

        if not expected_counts.equals(actual_counts):
            raise ValueError(
                f"24-hour observation count validation failed for {station_id}."
            )

    # A 24-hour rolling mean must not exist with fewer than 18 readings.
    invalid_rolling_values = (
        feature_df["pm25_rolling_mean_24h"].notna()
        & (
            feature_df["pm25_rolling_observation_count_24h"] < 18
        )
    )

    if invalid_rolling_values.any():
        raise ValueError(
            "24-hour rolling mean exists with fewer than 18 observations."
        )

    # Every eligible window must have a calculated rolling mean.
    missing_eligible_means = (
        feature_df["pm25_rolling_mean_24h"].isna()
        & (
            feature_df["pm25_rolling_observation_count_24h"] >= 18
        )
    )

    if missing_eligible_means.any():
        raise ValueError(
            "Eligible 24-hour rolling means are unexpectedly missing."
        )


    print("Feature quality checks passed.")


# --------------------------------------------------
# 7. MAIN PIPELINE
# --------------------------------------------------

def main():

    print("Loading hourly panel...")
    hourly_df = load_hourly_panel()

    print("Validating hourly panel...")
    validate_hourly_panel(hourly_df)

    print("\nBuilding feature dataset...")
    feature_df = build_feature_dataset(hourly_df)

    print("\nValidating feature dataset...")
    validate_feature_dataset(hourly_df, feature_df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFEATURE ENGINEERING COMPLETED")
    print("-" * 45)

    print("Total rows:", len(feature_df))
    print("Total columns:", len(feature_df.columns))

    print(
        "Missing current PM2.5:",
        feature_df["pm25_at_origin"].isna().sum()
    )

    print(
        "Missing next-hour targets:",
        feature_df["target_pm25_next_hour"].isna().sum()
    )

    print(
        "Valid next-hour targets:",
        feature_df["target_pm25_next_hour"].notna().sum()
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
