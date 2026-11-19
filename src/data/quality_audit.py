
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_FILES = {
    "TRAIN": PROCESSED_DIR / "train.csv",
    "VALIDATION": PROCESSED_DIR / "validation.csv",
    "TEST": PROCESSED_DIR / "test.csv",
}

TARGET_COLUMN = "target_pm25_next_hour"

ROLLING_MEAN_COLUMN = "pm25_rolling_mean_24h"
ROLLING_COUNT_COLUMN = "pm25_rolling_observation_count_24h"

MIN_ROLLING_OBSERVATIONS = 18
ROLLING_WINDOW_SIZE = 24

NON_FEATURE_COLUMNS = {
    "period_start",
    "target_timestamp",
    "is_observed",
    "target_is_observed",
    TARGET_COLUMN,
}


# --------------------------------------------------
# 2. VALIDATE ROLLING FEATURE POLICY
# --------------------------------------------------

def validate_rolling_policy(df, split_name):

    required_columns = {
        ROLLING_MEAN_COLUMN,
        ROLLING_COUNT_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name}: Missing rolling columns: {missing_columns}"
        )

    counts = pd.to_numeric(
        df[ROLLING_COUNT_COLUMN],
        errors="coerce",
    )

    if counts.isna().any():
        raise ValueError(
            f"{split_name}: Rolling observation counts contain invalid values."
        )

    if not np.isfinite(counts).all():
        raise ValueError(
            f"{split_name}: Rolling observation counts contain non-finite values."
        )

    if not counts.between(0, ROLLING_WINDOW_SIZE).all():
        raise ValueError(
            f"{split_name}: Rolling counts must be between 0 and 24."
        )

    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError(
            f"{split_name}: Rolling counts must be integers."
        )

    rolling_mean = df[ROLLING_MEAN_COLUMN]

    # A rolling mean must not exist below the 18-observation threshold.
    invalid_means = (
        rolling_mean.notna()
        & (counts < MIN_ROLLING_OBSERVATIONS)
    )

    if invalid_means.any():
        raise ValueError(
            f"{split_name}: Rolling mean exists with fewer than "
            f"{MIN_ROLLING_OBSERVATIONS} observations."
        )

    # Every eligible rolling window must have a calculated mean.
    missing_eligible_means = (
        rolling_mean.isna()
        & (counts >= MIN_ROLLING_OBSERVATIONS)
    )

    if missing_eligible_means.any():
        raise ValueError(
            f"{split_name}: Eligible rolling means are unexpectedly missing."
        )

    print("\n24-hour rolling policy validation: PASSED")
    print("Minimum required observations:", MIN_ROLLING_OBSERVATIONS)
    print("Maximum window observations:", ROLLING_WINDOW_SIZE)
    print("Rows with 18-24 observations:", int(counts.ge(18).sum()))
    print("Rows below the threshold:", int(counts.lt(18).sum()))


# --------------------------------------------------
# 3. VALIDATE TARGET INTEGRITY
# --------------------------------------------------

def validate_target_integrity(df, split_name):

    required_columns = {
        "target_timestamp",
        "period_start",
        "target_is_observed",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name}: Missing target metadata: {missing_columns}"
        )

    origin = pd.to_datetime(
        df["period_start"],
        errors="raise",
    )

    target_time = pd.to_datetime(
        df["target_timestamp"],
        errors="raise",
    )

    expected_target_time = origin + pd.Timedelta(hours=1)

    if not target_time.equals(expected_target_time):
        raise ValueError(
            f"{split_name}: Target timestamps are not exactly one hour "
            "after their prediction origins."
        )

    target_observed = df["target_is_observed"].astype(bool)
    target_available = df[TARGET_COLUMN].notna()

    if not target_observed.equals(target_available):
        raise ValueError(
            f"{split_name}: Target observation flags do not match "
            "target value availability."
        )

    print("\nTarget integrity validation: PASSED")
    print("Target timestamp alignment: exact next hour")
    print("Target observation flags: consistent")


# --------------------------------------------------
# 4. AUDIT ONE DATASET SPLIT
# --------------------------------------------------

def audit_split(name, file_path):

    print("\n" + "=" * 65)
    print(f"{name} DATA QUALITY AUDIT")
    print("=" * 65)

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{name}: Dataset is empty.")

    required_columns = {
        "station_id",
        "period_start",
        "target_timestamp",
        "is_observed",
        "target_is_observed",
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name}: Missing required columns: {missing_columns}"
        )

    if df.duplicated(["station_id", "period_start"]).any():
        raise ValueError(
            f"{name}: Duplicate station-hour records found."
        )

    # ----------------------------------------------
    # A. VALIDATE FEATURE AND TARGET POLICIES
    # ----------------------------------------------

    validate_rolling_policy(df, name)
    validate_target_integrity(df, name)

    # ----------------------------------------------
    # B. IDENTIFY MODEL INPUT FEATURES
    # ----------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model input features found.")

    missing_counts = df[feature_columns].isna().sum()

    complete_inputs = df[feature_columns].notna().all(axis=1)

    valid_target = df[TARGET_COLUMN].notna()

    usable_rows = complete_inputs & valid_target

    # ----------------------------------------------
    # C. DATASET OVERVIEW
    # ----------------------------------------------

    print("\nDataset overview")
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))
    print("Model input features:", len(feature_columns))

    print("\nModel input features")

    for column in feature_columns:
        print("-", column)

    # ----------------------------------------------
    # D. TARGET AVAILABILITY
    # ----------------------------------------------

    print("\nTarget availability")
    print("Valid targets:", int(valid_target.sum()))
    print("Missing targets:", int((~valid_target).sum()))

    # ----------------------------------------------
    # E. INPUT FEATURE COMPLETENESS
    # ----------------------------------------------

    print("\nInput feature completeness")
    print("Rows with all input features:", int(complete_inputs.sum()))
    print(
        "Rows with at least one missing input:",
        int((~complete_inputs).sum()),
    )

    print("\nOverall supervised-learning usability")
    print(
        "Rows with complete inputs AND valid target:",
        int(usable_rows.sum()),
    )

    print(
        "Usable percentage:",
        f"{usable_rows.mean() * 100:.2f}%",
    )

    # ----------------------------------------------
    # F. MISSING VALUES PER FEATURE
    # ----------------------------------------------

    print("\nMissing values per input feature")
    print(missing_counts.to_string())

    # ----------------------------------------------
    # G. STATION-WISE AUDIT
    # ----------------------------------------------

    audit_data = df.assign(
        _valid_target=valid_target,
        _complete_inputs=complete_inputs,
        _usable_rows=usable_rows,
    )

    station_audit = audit_data.groupby("station_id").agg(
        total_rows=("station_id", "size"),
        valid_targets=("_valid_target", "sum"),
        complete_inputs=("_complete_inputs", "sum"),
        usable_rows=("_usable_rows", "sum"),
    )

    print("\nStation-wise audit")
    print(station_audit.to_string())

    print(f"\n{name} audit completed successfully.")

    return {
        "split": name,
        "total_rows": len(df),
        "valid_targets": int(valid_target.sum()),
        "complete_inputs": int(complete_inputs.sum()),
        "usable_rows": int(usable_rows.sum()),
    }


# --------------------------------------------------
# 5. MAIN QUALITY GATE
# --------------------------------------------------

def main():

    print("AIRPULSE — STAGE 4E: DATA QUALITY GATE")
    print("Auditing train, validation, and test datasets.")

    results = []

    for name, file_path in SPLIT_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        result = audit_split(name, file_path)
        results.append(result)

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    summary = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL QUALITY GATE SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nTotal usable supervised rows:",
          summary["usable_rows"].sum())

    print("\nDATA QUALITY AUDIT COMPLETED")
    print("No data was modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
