"""
tests/test_monitoring.py
------------------------
Automated tests for Model Monitoring and Retraining Workflow (Tasks A - D).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from monitoring.quality_gate import evaluate_gate, TOLERANCE
from src.models.residual_hgb import EXPECTED_FEATURES, TARGET_COLUMN, build_residual_hgb_pipeline


def test_task_a_schema_validation():
    """Verify that Task A checks for all 18 features plus target column."""
    required = EXPECTED_FEATURES + [TARGET_COLUMN]
    assert len(EXPECTED_FEATURES) == 18
    assert len(required) == 19
    assert "station_id" in required
    assert "pm25_at_origin" in required
    assert "target_pm25_next_hour" in required


def test_task_a_clean_batch_integrity():
    """Verify clean new_batch.csv meets schema, no nulls, and no duplicates."""
    repo_root = Path(__file__).resolve().parent.parent
    batch_file = repo_root / "monitoring" / "data" / "new_batch.csv"
    if not batch_file.exists():
        pytest.skip("new_batch.csv not generated yet")

    df = pd.read_csv(str(batch_file))
    required = EXPECTED_FEATURES + [TARGET_COLUMN]

    for col in required:
        assert col in df.columns, f"Missing required column: {col}"

    assert df[required].isnull().sum().sum() == 0, "Clean batch must contain zero null values"
    assert df.duplicated().sum() == 0, "Clean batch must contain zero duplicate rows"
    assert len(df) > 0, "Batch must not be empty"


def test_task_b_evaluation_report_schema():
    """Verify Task B evaluation report exists and contains required metrics."""
    repo_root = Path(__file__).resolve().parent.parent
    report_file = repo_root / "monitoring" / "reports" / "evaluation_report.json"
    if not report_file.exists():
        pytest.skip("evaluation_report.json not generated yet")

    with open(str(report_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metrics" in data
    assert "mae" in data["metrics"]
    assert "rmse" in data["metrics"]
    assert "r2" in data["metrics"]
    assert data["metrics"]["mae"] > 0
    assert data["metrics"]["rmse"] > 0
    assert "persistence_baseline_metrics" in data
    assert "station_breakdown" in data


def test_task_c_candidate_pipeline_fit_predict():
    """Verify candidate pipeline can fit on small combined data and predict finite values."""
    pipeline = build_residual_hgb_pipeline()
    n = 30
    rng = np.random.RandomState(42)

    X_dummy = pd.DataFrame({
        "station_id": ["site_113", "site_115", "site_117"] * 10,
        "pm25_at_origin": rng.uniform(20, 100, n),
        "pm25_1h_before": rng.uniform(20, 100, n),
        "pm25_2h_before": rng.uniform(20, 100, n),
        "pm25_23h_before": rng.uniform(20, 100, n),
        "pm25_rolling_mean_3h": rng.uniform(20, 100, n),
        "pm25_rolling_mean_6h": rng.uniform(20, 100, n),
        "pm25_rolling_mean_24h": rng.uniform(20, 100, n),
        "pm25_rolling_observation_count_24h": [24] * n,
        "hour": rng.randint(0, 24, n),
        "day_of_week": rng.randint(0, 7, n),
        "month": [11] * n,
        "hour_sin": np.sin(2 * np.pi * rng.randint(0, 24, n) / 24),
        "hour_cos": np.cos(2 * np.pi * rng.randint(0, 24, n) / 24),
        "day_of_week_sin": [0.0] * n,
        "day_of_week_cos": [1.0] * n,
        "month_sin": [-0.86] * n,
        "month_cos": [0.5] * n,
    })
    y_dummy = X_dummy["pm25_at_origin"].to_numpy() + rng.normal(0, 3, n)

    pipeline.fit(X_dummy, y_dummy)
    preds = pipeline.predict(X_dummy)

    assert len(preds) == n
    assert np.isfinite(preds).all()


def test_task_d_quality_gate_promote():
    """Verify quality gate marks candidate as PROMOTE when candidate MAE is better or within tolerance."""
    champ_m = {"mae": 10.0, "rmse": 20.0, "r2": 0.85}
    cand_m  = {"mae": 8.5,  "rmse": 18.0, "r2": 0.88}

    res = evaluate_gate(champ_m, cand_m, "champ.joblib", "cand.joblib", tolerance=1.02)
    assert res["decision"] == "PROMOTE"
    assert "eligible for promotion" in res["reason"]


def test_task_d_quality_gate_retain():
    """Verify quality gate retains champion when candidate MAE degrades beyond tolerance."""
    champ_m = {"mae": 10.0, "rmse": 20.0, "r2": 0.85}
    cand_m  = {"mae": 15.0, "rmse": 25.0, "r2": 0.70}

    res = evaluate_gate(champ_m, cand_m, "champ.joblib", "cand.joblib", tolerance=1.02)
    assert res["decision"] == "RETAIN"
    assert "Champion is retained" in res["reason"]
