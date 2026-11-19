
import pandas as pd
import pytest

from src.data.preprocess import (
    load_raw_data,
    validate_raw_data,
    build_hourly_panel,
    validate_processed_data,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def processed_df(raw_df):
    validate_raw_data(raw_df)
    return build_hourly_panel(raw_df)


def test_raw_dataset_integrity(raw_df):
    assert len(raw_df) == 25283
    assert raw_df["station_id"].nunique() == 3
    assert raw_df["mean"].notna().all()
    assert (raw_df["mean"] >= 0).all()
    assert not raw_df.duplicated(
        ["station_id", "period_start"]
    ).any()


def test_processed_panel_dimensions(processed_df):
    assert len(processed_df) == 26352
    assert processed_df["station_id"].nunique() == 3

    counts = processed_df.groupby("station_id").size()

    assert (counts == 8784).all()


def test_missing_observations_are_preserved(processed_df):
    missing = processed_df[
        ~processed_df["is_observed"]
    ]

    assert len(missing) == 1069
    assert missing["mean"].isna().all()

    assert int(
        (processed_df["is_observed"]).sum()
    ) == 25283


def test_station_missing_hour_counts(processed_df):
    missing_counts = (
        processed_df.groupby("station_id")["is_observed"]
        .apply(lambda x: int((~x).sum()))
        .to_dict()
    )

    assert missing_counts == {
        "site_113": 283,
        "site_115": 347,
        "site_117": 439,
    }


def test_original_values_are_preserved(raw_df, processed_df):
    validate_processed_data(processed_df, raw_df)


def test_processed_timestamps_are_hourly(processed_df):
    for station_id, group in processed_df.groupby("station_id"):
        timestamps = group["period_start"].sort_values()

        differences = timestamps.diff().dropna()

        assert (differences == pd.Timedelta(hours=1)).all(), (
            f"Hourly timeline broken for {station_id}"
        )
