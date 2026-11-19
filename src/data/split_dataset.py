
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

TIMEZONE = "Asia/Kolkata"


# --------------------------------------------------
# 2. LOAD FEATURE DATASET
# --------------------------------------------------

def load_feature_dataset():

    df = pd.read_csv(INPUT_FILE)

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="raise"
    )

    df["target_timestamp"] = pd.to_datetime(
        df["target_timestamp"],
        errors="raise"
    )

    if df["target_timestamp"].dt.tz is None:
        raise ValueError("Target timezone information is missing.")

    df = df.sort_values(
        ["target_timestamp", "station_id"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# 3. CHRONOLOGICAL SPLITTING
# --------------------------------------------------

def split_dataset(df):

    train_end = pd.Timestamp(
        "2024-09-01 00:00:00",
        tz=TIMEZONE
    )

    validation_end = pd.Timestamp(
        "2024-11-01 00:00:00",
        tz=TIMEZONE
    )

    train_df = df[
        df["target_timestamp"] < train_end
    ].copy()

    validation_df = df[
        (df["target_timestamp"] >= train_end)
        & (df["target_timestamp"] < validation_end)
    ].copy()

    test_df = df[
        df["target_timestamp"] >= validation_end
    ].copy()

    return train_df, validation_df, test_df


# --------------------------------------------------
# 4. VALIDATE SPLITS
# --------------------------------------------------

def validate_splits(
    original_df,
    train_df,
    validation_df,
    test_df
):

    # No rows should be lost or duplicated.
    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_rows != len(original_df):
        raise ValueError("Split row count mismatch.")

    # All splits must be non-empty.
    if any(
        split.empty
        for split in [train_df, validation_df, test_df]
    ):
        raise ValueError("One or more splits are empty.")

    # Validate chronological boundaries.
    if not (
        train_df["target_timestamp"].max()
        < validation_df["target_timestamp"].min()
    ):
        raise ValueError("Training and validation overlap.")

    if not (
        validation_df["target_timestamp"].max()
        < test_df["target_timestamp"].min()
    ):
        raise ValueError("Validation and test overlap.")

    # Verify every original row appears exactly once.
    combined = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True
    )

    key_columns = ["station_id", "period_start"]

    if combined.duplicated(key_columns).any():
        raise ValueError("Duplicate rows found across splits.")

    if set(
        map(tuple, combined[key_columns].to_numpy())
    ) != set(
        map(tuple, original_df[key_columns].to_numpy())
    ):
        raise ValueError("Original records are missing from splits.")

    print("Chronological split validation passed.")


# --------------------------------------------------
# 5. SAVE SPLITS
# --------------------------------------------------

def main():

    print("Loading feature dataset...")

    df = load_feature_dataset()

    print("Total records:", len(df))

    print("\nCreating chronological splits...")

    train_df, validation_df, test_df = split_dataset(df)

    print("\nValidating splits...")

    validate_splits(
        df,
        train_df,
        validation_df,
        test_df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nCHRONOLOGICAL SPLITTING COMPLETED")
    print("-" * 45)

    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Testing rows:", len(test_df))

    print("\nSaved files:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()
