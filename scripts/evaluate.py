"""Convenience evaluation entrypoint.

Runs validation evaluation for the residual HGB model against persistence
and logs results to the MLflow tracking server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

from src.models.evaluate_residual_hgb import main

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


if __name__ == "__main__":
    with mlflow.start_run(run_name="validation_evaluation"):
        main()
