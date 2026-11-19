
import pandas as pd
import pytest

from src.data import prepare_model_data as model_data


def test_load_split_raises_error_for_missing_file(tmp_path):
    """A missing source file should raise FileNotFoundError."""

    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        model_data.load_split(file_path, "train")


def test_load_split_raises_error_for_empty_file(tmp_path):
    """An empty CSV should raise ValueError."""

    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    with pytest.raises(ValueError):
        model_data.load_split(file_path, "train")


def test_load_split_rejects_missing_required_columns(tmp_path):
    """A CSV missing required model columns should be rejected."""

    file_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"station_id": ["site_113"]}).to_csv(
        file_path,
        index=False,
    )

    with pytest.raises(ValueError, match="Missing columns"):
        model_data.load_split(file_path, "train")


def test_load_split_rejects_duplicate_station_origin_records(tmp_path):
    """Duplicate station-origin combinations should be rejected."""

    data = {
        "station_id": ["site_113", "site_113"],
        "period_start": [
            "2024-01-01 00:00:00+05:30",
            "2024-01-01 00:00:00+05:30",
        ],
    }

    for column in (
        model_data.INPUT_FEATURES
        + [model_data.TARGET_COLUMN]
        + model_data.METADATA_COLUMNS
    ):
        if column not in data:
            data[column] = [1, 1]

    file_path = tmp_path / "duplicates.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Duplicate"):
        model_data.load_split(file_path, "train")


@pytest.fixture
def sample_split_df():
    """Create a small controlled dataset for model preparation tests."""

    rows = []

    for i in range(4):
        row = {
            "station_id": "site_113",
            "period_start": pd.Timestamp("2024-01-01 00:00:00")
            + pd.Timedelta(hours=i),
            "target_timestamp": pd.Timestamp("2024-01-01 01:00:00")
            + pd.Timedelta(hours=i),
            "is_observed": True,
            "target_is_observed": True,
            "target_pm25_next_hour": 50.0 + i,
        }

        for feature in model_data.INPUT_FEATURES:
            if feature == "station_id":
                row[feature] = "site_113"
            else:
                row[feature] = float(i + 1)

        rows.append(row)

    return pd.DataFrame(rows)


def test_prepare_split_returns_correct_columns(sample_split_df, monkeypatch):
    """Inputs, targets and metadata should have separate columns."""

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        4,
    )

    X, y, metadata = model_data.prepare_split(
        sample_split_df,
        "train",
    )

    assert X.columns.tolist() == model_data.INPUT_FEATURES
    assert y.name == model_data.TARGET_COLUMN
    assert metadata.columns.tolist() == model_data.METADATA_COLUMNS


def test_prepare_split_excludes_rows_with_missing_inputs(
    sample_split_df,
    monkeypatch,
):
    """Rows with incomplete input features should be excluded."""

    sample_split_df.loc[0, "pm25_1h_before"] = float("nan")

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        3,
    )

    X, y, metadata = model_data.prepare_split(
        sample_split_df,
        "train",
    )

    assert len(X) == 3
    assert len(y) == 3
    assert len(metadata) == 3
    assert not X.isna().any().any()

def test_prepare_split_excludes_rows_with_missing_targets(
    sample_split_df,
    monkeypatch,
):
    """Rows with missing targets should be excluded."""

    sample_split_df.loc[0, model_data.TARGET_COLUMN] = float("nan")
    sample_split_df.loc[0, "target_is_observed"] = False

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        3,
    )

    X, y, metadata = model_data.prepare_split(
        sample_split_df,
        "train",
    )

    assert len(X) == 3
    assert len(y) == 3
    assert len(metadata) == 3
    assert not y.isna().any()

def test_prepare_split_rejects_incorrect_target_timestamp(
    sample_split_df,
    monkeypatch,
):
    """Target timestamps must be exactly one hour after origin."""

    sample_split_df.loc[0, "target_timestamp"] += pd.Timedelta(hours=1)

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        4,
    )

    with pytest.raises(ValueError, match="alignment"):
        model_data.prepare_split(sample_split_df, "train")

def test_prepare_split_rejects_inconsistent_target_observation_flag(
    sample_split_df,
    monkeypatch,
):
    """Observed-target flags must agree with target availability."""

    sample_split_df.loc[0, "target_is_observed"] = False

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        4,
    )

    with pytest.raises(ValueError, match="observation flags"):
        model_data.prepare_split(sample_split_df, "train")

def test_prepare_split_rejects_missing_target_marked_as_observed(
    sample_split_df,
    monkeypatch,
):
    """A missing target must not be marked as observed."""

    sample_split_df.loc[0, model_data.TARGET_COLUMN] = float("nan")
    sample_split_df.loc[0, "target_is_observed"] = True

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        3,
    )

    with pytest.raises(ValueError, match="observation flags"):
        model_data.prepare_split(sample_split_df, "train")

def test_prepare_split_rejects_unexpected_row_count(
    sample_split_df,
    monkeypatch,
):
    """Unexpected prepared row counts should raise ValueError."""

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        5,
    )

    with pytest.raises(ValueError, match="Expected 5 usable rows"):
        model_data.prepare_split(sample_split_df, "train")


def test_save_split_creates_correct_files_and_preserves_data(
    sample_split_df,
    monkeypatch,
    tmp_path,
):
    """Save inputs, targets and metadata as separate CSV files."""

    monkeypatch.setitem(
        model_data.EXPECTED_USABLE_ROWS,
        "train",
        4,
    )

    monkeypatch.setattr(model_data, "OUTPUT_DIR", tmp_path)

    X, y, metadata = model_data.prepare_split(
        sample_split_df,
        "train",
    )

    model_data.save_split(X, y, metadata, "train")

    X_file = tmp_path / "X_train.csv"
    y_file = tmp_path / "y_train.csv"
    metadata_file = tmp_path / "metadata_train.csv"

    assert X_file.exists()
    assert y_file.exists()
    assert metadata_file.exists()

    saved_X = pd.read_csv(X_file)
    saved_y = pd.read_csv(y_file)
    saved_metadata = pd.read_csv(metadata_file)

    assert saved_X.columns.tolist() == model_data.INPUT_FEATURES
    assert saved_y.columns.tolist() == [model_data.TARGET_COLUMN]
    assert saved_metadata.columns.tolist() == model_data.METADATA_COLUMNS

    pd.testing.assert_frame_equal(saved_X, X)
    pd.testing.assert_series_equal(
        saved_y[model_data.TARGET_COLUMN],
        y,
        check_dtype=False,
        check_names=True,
    )
    assert len(saved_metadata) == len(metadata)
