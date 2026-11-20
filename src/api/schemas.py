"""Pydantic request/response schemas for the AirPulse prediction API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.model_loader import get_feature_columns, get_anchor_column
from src.mlflow_config import REGISTERED_MODEL_NAME

_FEATURE_COLUMNS = get_feature_columns()
_ANCHOR_COLUMN = get_anchor_column()


class PredictionRequest(BaseModel):
    """Request payload for a single next-hour PM2.5 prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    station_id: str = Field(
        ...,
        description="Monitoring station ID (site_113, site_115, or site_117).",
        examples=["site_113"],
    )
    pm25_at_origin: float = Field(
        ..., description="Current PM2.5 concentration at the station (µg/m³).", ge=0
    )
    pm25_1h_before: float = Field(..., ge=0)
    pm25_2h_before: float = Field(..., ge=0)
    pm25_23h_before: float = Field(..., ge=0)
    pm25_rolling_mean_3h: float = Field(..., ge=0)
    pm25_rolling_mean_6h: float = Field(..., ge=0)
    pm25_rolling_mean_24h: float = Field(..., ge=0)
    pm25_rolling_observation_count_24h: float = Field(..., ge=0, le=24)
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23).")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday).")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12).")
    hour_sin: float = Field(..., ge=-1.0, le=1.0)
    hour_cos: float = Field(..., ge=-1.0, le=1.0)
    day_of_week_sin: float = Field(..., ge=-1.0, le=1.0)
    day_of_week_cos: float = Field(..., ge=-1.0, le=1.0)
    month_sin: float = Field(..., ge=-1.0, le=1.0)
    month_cos: float = Field(..., ge=-1.0, le=1.0)

    @field_validator("station_id")
    @classmethod
    def validate_station_id(cls, v: str) -> str:
        allowed = {"site_113", "site_115", "site_117"}
        if v not in allowed:
            raise ValueError(f"station_id must be one of {sorted(allowed)}")
        return v

    def to_dataframe(self) -> Any:
        """Convert the request to a pandas DataFrame with the expected feature order."""
        import pandas as pd

        data: dict[str, list[Any]] = {}
        for col in _FEATURE_COLUMNS:
            value = getattr(self, col)
            data[col] = [value]
        return pd.DataFrame(data)


class PredictionResponse(BaseModel):
    """Response payload for a next-hour PM2.5 prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "station_id": "site_113",
                "predicted_pm25_next_hour": 90.91,
                "anchor_pm25_at_origin": 97.0,
                "residual": -6.09,
                "model": "Persistence-Anchored Residual HistGradientBoosting",
                "forecast_horizon": "1_hour",
                "timestamp_utc": "2026-11-18T00:00:00+00:00",
            }
        }
    )

    station_id: str
    predicted_pm25_next_hour: float
    anchor_pm25_at_origin: float
    residual: float
    model: str = "Persistence-Anchored Residual HistGradientBoosting"
    forecast_horizon: str = "1_hour"
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    model_artifact: str = f"models:/{REGISTERED_MODEL_NAME}@champion"
    model_source: str | None = None