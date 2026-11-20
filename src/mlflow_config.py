"""Shared MLflow configuration for the AirPulse tracking workflow."""

import os


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:7001",
)
MLFLOW_EXPERIMENT_NAME = "AirPulse_PM25_Forecasting"
REGISTERED_MODEL_NAME = "airpulse-pm25-forecast"
