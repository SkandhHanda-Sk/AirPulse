import numpy as np
import pandas as pd

from src.features.build_features import (
    build_station_features,
    build_feature_dataset,
    validate_feature_dataset,
)


def create_sample_panel():
    """Create a two-station hourly dataset with one missing observation."""

    timestamps = pd.date_range(
        start="2024-01-01 00:00",
        periods=30,
        freq="h",
        tz="Asia/Kolkata",
    )

    rows = []

    for station in ["site_113", "site_115"]:
        offset = 0 if station == "site_113" else 100

        for i, timestamp in enumerate(timestamps):
            value = float(i + 10 + offset)

            if i == 5:
                value = np.nan

            rows.append({
                "station_id": station,
                "parameter_name": "PM2.5",
                "unit": "µg/m³",
                "period_start": timestamp,
                "mean": value,
                "min": value,
                "max": value,
                "n": 1,
                "is_observed": not pd.isna(value),
            })

    return pd.DataFrame(rows)


def test_next_hour_target_alignment():
    """Target must refer to exactly the next hour."""

    panel = create_sample_panel()

    features = build_feature_dataset(panel)

    station = features[
        features["station_id"] == "site_113"
    ].reset_index(drop=True)

    assert station.loc[0, "target_timestamp"] == (
        station.loc[0, "period_start"] + pd.Timedelta(hours=1)
    )

    assert station.loc[0, "target_pm25_next_hour"] == 11.0


def test_missing_next_hour_is_not_skipped():
    """A missing next-hour observation must produce a missing target."""

    panel = create_sample_panel()

    features = build_station_features(
        panel[panel["station_id"] == "site_113"]
    )

    features = features.reset_index(drop=True)

    # Origin hour 4 targets hour 5, which is missing.
    assert pd.isna(features.loc[4, "target_pm25_next_hour"])
    assert not features.loc[4, "target_is_observed"]

    # Origin hour 5 targets hour 6, which is valid.
    assert features.loc[5, "target_pm25_next_hour"] == 16.0
    assert features.loc[5, "target_is_observed"]


def test_lag_alignment():
    """Lag features must use exact previous hourly observations."""

    panel = create_sample_panel()

    features = build_station_features(
        panel[panel["station_id"] == "site_113"]
    ).reset_index(drop=True)

    assert features.loc[1, "pm25_1h_before"] == 10.0
    assert features.loc[2, "pm25_2h_before"] == 10.0

    # Hour 6 has a missing observation at hour 5 as its previous hour.
    assert pd.isna(features.loc[6, "pm25_1h_before"])

    # Hour 7's one-hour lag refers to hour 6.
    assert features.loc[7, "pm25_1h_before"] == 16.0

    # At hour 23, the 23-hour lag refers to hour 0.
    assert features.loc[23, "pm25_23h_before"] == 10.0



def test_rolling_mean_respects_observation_thresholds():
    """3h and 6h windows stay strict; 24h requires at least 18 observations."""

    panel = create_sample_panel()

    features = build_station_features(
        panel[panel["station_id"] == "site_113"]
    ).reset_index(drop=True)

    # Missing hour 5 makes 3-hour windows ending at 5, 6 and 7 incomplete.
    assert pd.isna(features.loc[5, "pm25_rolling_mean_3h"])
    assert pd.isna(features.loc[6, "pm25_rolling_mean_3h"])
    assert pd.isna(features.loc[7, "pm25_rolling_mean_3h"])

    # The 3-hour window ending at hour 8 is complete.
    assert features.loc[8, "pm25_rolling_mean_3h"] == 17.0

    # The 24-hour window ending at hour 23 has 23 valid observations.
    assert features.loc[23, "pm25_rolling_observation_count_24h"] == 23

    # It must now produce a mean under the 18/24 policy.
    assert pd.notna(features.loc[23, "pm25_rolling_mean_24h"])

    expected_mean = features.loc[:23, "pm25_at_origin"].mean()

    assert np.isclose(
        features.loc[23, "pm25_rolling_mean_24h"],
        expected_mean
    )

    # A window with fewer than 18 observations must not produce a mean.
    below_threshold = (
        features["pm25_rolling_observation_count_24h"] < 18
    )

    assert features.loc[
        below_threshold, "pm25_rolling_mean_24h"
    ].isna().all()


def test_station_features_do_not_mix():
    """Each station must have independently calculated features."""

    panel = create_sample_panel()

    features = build_feature_dataset(panel)

    station_113 = features[
        features["station_id"] == "site_113"
    ].reset_index(drop=True)

    station_115 = features[
        features["station_id"] == "site_115"
    ].reset_index(drop=True)

    assert station_113.loc[1, "pm25_at_origin"] == 11.0
    assert station_115.loc[1, "pm25_at_origin"] == 111.0

    assert station_113.loc[1, "pm25_1h_before"] == 10.0
    assert station_115.loc[1, "pm25_1h_before"] == 110.0


def test_cyclical_encoding_values():
    """Cyclical calendar encodings must match their expected formulas."""

    panel = create_sample_panel()

    features = build_station_features(
        panel[panel["station_id"] == "site_113"]
    ).reset_index(drop=True)

    # First timestamp is midnight on Monday in this sample.
    assert np.isclose(features.loc[0, "hour_sin"], 0.0)
    assert np.isclose(features.loc[0, "hour_cos"], 1.0)

    assert np.isclose(features.loc[0, "day_of_week_sin"], 0.0)
    assert np.isclose(features.loc[0, "day_of_week_cos"], 1.0)

    # January: month index is zero in the cyclical formula.
    assert np.isclose(features.loc[0, "month_sin"], 0.0)
    assert np.isclose(features.loc[0, "month_cos"], 1.0)


def test_feature_dataset_validation():
    """The feature dataset must pass its actual quality validator."""

    panel = create_sample_panel()

    features = build_feature_dataset(panel)

    # Should complete without raising a ValueError.
    validate_feature_dataset(panel, features)

    assert len(features) == len(panel)
    assert not features.duplicated(
        ["station_id", "period_start"]
    ).any()