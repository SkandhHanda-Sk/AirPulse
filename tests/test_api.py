"""Tests for the AirPulse FastAPI prediction API."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """GET /health should return 200 with model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_artifact" in data


def test_root_endpoint_returns_200():
    """GET / should return 200 with service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AirPulse PM2.5 Forecasting API"
    assert data["status"] == "online"
    assert "docs_url" in data


def test_predict_endpoint_returns_valid_prediction():
    """POST /predict should return a valid next-hour PM2.5 prediction."""
    payload = {
        "station_id": "site_113",
        "pm25_at_origin": 97.0,
        "pm25_1h_before": 95.0,
        "pm25_2h_before": 93.0,
        "pm25_23h_before": 88.0,
        "pm25_rolling_mean_3h": 94.0,
        "pm25_rolling_mean_6h": 92.0,
        "pm25_rolling_mean_24h": 90.0,
        "pm25_rolling_observation_count_24h": 24.0,
        "hour": 14,
        "day_of_week": 1,
        "month": 11,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "day_of_week_sin": 0.78,
        "day_of_week_cos": 0.62,
        "month_sin": 0.97,
        "month_cos": -0.24,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()
    assert data["station_id"] == "site_113"
    assert data["anchor_pm25_at_origin"] == 97.0
    assert isinstance(data["predicted_pm25_next_hour"], float)
    assert isinstance(data["residual"], float)
    assert data["model"] == "Persistence-Anchored Residual HistGradientBoosting"
    assert data["forecast_horizon"] == "1_hour"
    assert "timestamp_utc" in data


def test_predict_rejects_invalid_station_id():
    """POST /predict with invalid station_id should return 422."""
    payload = {
        "station_id": "invalid_station",
        "pm25_at_origin": 50.0,
        "pm25_1h_before": 48.0,
        "pm25_2h_before": 46.0,
        "pm25_23h_before": 44.0,
        "pm25_rolling_mean_3h": 47.0,
        "pm25_rolling_mean_6h": 46.0,
        "pm25_rolling_mean_24h": 45.0,
        "pm25_rolling_observation_count_24h": 24.0,
        "hour": 10,
        "day_of_week": 2,
        "month": 1,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "day_of_week_sin": 0.0,
        "day_of_week_cos": 1.0,
        "month_sin": 0.0,
        "month_cos": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_all_three_stations():
    """POST /predict should work for all three supported stations."""
    for station_id in ("site_113", "site_115", "site_117"):
        payload = {
            "station_id": station_id,
            "pm25_at_origin": 50.0,
            "pm25_1h_before": 48.0,
            "pm25_2h_before": 46.0,
            "pm25_23h_before": 44.0,
            "pm25_rolling_mean_3h": 47.0,
            "pm25_rolling_mean_6h": 46.0,
            "pm25_rolling_mean_24h": 45.0,
            "pm25_rolling_observation_count_24h": 24.0,
            "hour": 10,
            "day_of_week": 2,
            "month": 1,
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_of_week_sin": 0.0,
            "day_of_week_cos": 1.0,
            "month_sin": 0.0,
            "month_cos": 1.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Failed for {station_id}: {response.text}"
        data = response.json()
        assert data["station_id"] == station_id