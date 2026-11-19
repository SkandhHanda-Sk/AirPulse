
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "model_ready"

SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.csv",
    "validation": PROCESSED_DIR / "validation.csv",
    "test": PROCESSED_DIR / "test.csv",
}

INPUT_FEATURES = [
    "station_id",
    "pm25_at_origin",
    "pm25_1h_before",
    "pm25_2h_before",
    "pm25_23h_before",
    "pm25_rolling_mean_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_observation_count_24h",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

TARGET_COLUMN = "target_pm25_next_hour"

METADATA_COLUMNS = [
    "station_id",
    "period_start",
    "target_timestamp",
    "is_observed",
    "target_is_observed",
]

EXPECTED_USABLE_ROWS = {
    "train": 15016,
    "validation": 3629,
    "test": 3900,
}


# --------------------------------------------------
# 2. LOAD AND VALIDATE SOURCE DATA
# --------------------------------------------------

def load_split(file_path: Path, split_name: str) -> pd.DataFrame:
    """Load a split and validate its required columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{split_name}: Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{split_name}: Dataset is empty.")

    required_columns = (
        INPUT_FEATURES
        + [TARGET_COLUMN]
        + METADATA_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name}: Missing columns: {missing_columns}"
        )

    if df.duplicated(["station_id", "period_start"]).any():
        raise ValueError(
            f"{split_name}: Duplicate station-origin records found."
        )

    return df


# --------------------------------------------------
# 3. PREPARE MODEL DATA
# --------------------------------------------------

def prepare_split(
    df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create complete-case inputs, targets and metadata."""

    # Verify target observation flags against actual target availability.
    target_observed = (
        df["target_is_observed"].astype(bool)
    )

    if not target_observed.equals(df[TARGET_COLUMN].notna()):
        raise ValueError(
            f"{split_name}: Target observation flags are inconsistent."
        )

    # Identify rows with complete inputs and valid targets.
    complete_inputs = df[INPUT_FEATURES].notna().all(axis=1)
    valid_targets = df[TARGET_COLUMN].notna()

    usable_mask = complete_inputs & valid_targets
    prepared_df = df.loc[usable_mask].copy()

    if prepared_df.empty:
        raise ValueError(
            f"{split_name}: No usable supervised-learning rows."
        )

    X = prepared_df[INPUT_FEATURES].copy()
    y = prepared_df[TARGET_COLUMN].copy()
    metadata = prepared_df[METADATA_COLUMNS].copy()

    # Verify that no missing values remain.
    if X.isna().any().any():
        raise ValueError(
            f"{split_name}: Missing values remain in model inputs."
        )

    if y.isna().any():
        raise ValueError(
            f"{split_name}: Missing target values remain."
        )

    # Verify exact next-hour target alignment.
    origin = pd.to_datetime(
        metadata["period_start"],
        errors="raise",
    )

    target_time = pd.to_datetime(
        metadata["target_timestamp"],
        errors="raise",
    )

    if not target_time.equals(
        origin + pd.Timedelta(hours=1)
    ):
        raise ValueError(
            f"{split_name}: Target timestamp alignment failed."
        )

    # Verify target observation flags.
    target_observed = (
        metadata["target_is_observed"].astype(bool)
    )

    if not target_observed.equals(y.notna()):
        raise ValueError(
            f"{split_name}: Target observation flags are inconsistent."
        )

    # Verify expected usable row count.
    expected_rows = EXPECTED_USABLE_ROWS[split_name]

    if len(X) != expected_rows:
        raise ValueError(
            f"{split_name}: Expected {expected_rows} usable rows, "
            f"found {len(X)}."
        )

    return X.reset_index(drop=True), y.reset_index(drop=True), metadata.reset_index(drop=True)


# --------------------------------------------------
# 4. SAVE PREPARED DATASETS
# --------------------------------------------------

def save_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    split_name: str,
) -> None:
    """Save model inputs, targets and metadata separately."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X.to_csv(
        OUTPUT_DIR / f"X_{split_name}.csv",
        index=False,
    )

    y.to_frame(name=TARGET_COLUMN).to_csv(
        OUTPUT_DIR / f"y_{split_name}.csv",
        index=False,
    )

    metadata.to_csv(
        OUTPUT_DIR / f"metadata_{split_name}.csv",
        index=False,
    )


# --------------------------------------------------
# 5. MAIN PIPELINE
# --------------------------------------------------

def main() -> None:
    print("AIRPULSE — STAGE 5B: MODEL DATA PREPARATION")
    print("=" * 60)

    for split_name, file_path in SPLIT_FILES.items():

        print(f"\nPreparing {split_name.upper()} dataset...")

        df = load_split(file_path, split_name)

        X, y, metadata = prepare_split(df, split_name)

        save_split(X, y, metadata, split_name)

        print("Original rows:", len(df))
        print("Prepared rows:", len(X))
        print("Excluded rows:", len(df) - len(X))
        print("Input features:", X.shape[1])
        print("Target values:", len(y))
        print("Metadata rows:", len(metadata))

        print("\nStation-wise prepared rows:")
        print(X["station_id"].value_counts().sort_index().to_string())

    print("\n" + "=" * 60)
    print("MODEL DATA PREPARATION COMPLETED")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
