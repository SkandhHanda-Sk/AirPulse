# AirPulse — Project Brief

## Overview

AirPulse is an academic MLOps application for forecasting continuous PM2.5
concentration one hour ahead at three Delhi monitoring stations. The source is
the 2024 XKDR Forum India Air Quality Database. The model target is
`target_pm25_next_hour` in µg/m³, not AQI. AirPulse is not an official forecast
or health-advice service.

| Station ID | Location |
|---|---|
| `site_113` | Shadipur |
| `site_115` | NSIT Dwarka |
| `site_117` | ITO |

## Data and model

The pipeline constructs historical lag, rolling, and calendar features and
splits chronologically by target time. The existing 18-feature schema,
preprocessing, and train/validation/test methodology are preserved. Numeric
features pass through unchanged; `station_id` is imputed and one-hot encoded.
The preprocessing pipeline is fit on training data only.

The baseline predicts persistence (`pm25_at_origin`). The candidate is a
persistence-anchored residual HistGradientBoosting regressor:

```text
residual   = actual_next_hour_pm25 - pm25_at_origin
prediction = pm25_at_origin + predicted_residual
```

Current matched validation results (3,629 rows):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 10.7629 | 20.6201 | 0.8652 |
| Residual HGB | 10.2357 | 20.4997 | 0.8668 |

The saved test report is historical and predates the current artifact. Test
data has already been inspected and must not be used for tuning or another
evaluation run. A new untouched holdout period is needed for a future final
assessment.

## Application and MLOps

- **API:** FastAPI (`main.py`, `src/api/`), with `/health`, `/predict`, and
  OpenAPI docs at `/docs`.
- **Dashboard:** React, TypeScript, Vite (`frontend/`). It submits demonstration
  inputs; it is not connected to live station measurements. Historical charts
  are placeholders.
- **Experiment tracking:** MLflow 2.14.1, SQLite backend store (`mlflow.db`),
  artifact directory (`mlartifacts/`), tracking server at port 7001.
- **Experiment:** `AirPulse_PM25_Forecasting`.
- **Registered model:** `airpulse-pm25-forecast` (existing project name).
- **Serving URI:** `models:/airpulse-pm25-forecast@champion`.
- **Quality gate:** validation MAE must be lower than persistence MAE on the
  same validation rows. There is no arbitrary absolute cutoff. Training creates
  a new model version; only a gate-passing version receives the `champion`
  alias. A failed candidate leaves the current champion unchanged.
- **Docker:** the FastAPI image connects to host MLflow using
  `host.docker.internal:7001`. Use a Docker image built from the intended
  source revision; running an older image does not include subsequent source
  changes.

## Run locally (Windows CMD)

Replace `C:\path\to\AirPulse` in each CMD window with the extracted project
folder. Start the MLflow server in one window and leave it open:

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 7001 --default-artifact-root ./mlartifacts
```

In another CMD window, train and evaluate:

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python train.py
python -m scripts.evaluate
```

Start the current source API in a third window:

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in a fourth window:

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%\frontend"
npm run dev -- --host 127.0.0.1
```

Open the MLflow UI at `http://127.0.0.1:7001`, API docs at
`http://127.0.0.1:8000/docs`, and the Vite dashboard at
`http://127.0.0.1:5173`. If using the Docker API instead of local Uvicorn, do
not start both because both use port 8000.

## Repository map

```text
artifacts/models/  Local model output and metadata
backend/           Compatibility entrypoint
data/              Dataset and generated evaluation data
docs/              Supporting project documentation
frontend/          React dashboard and Vite configuration
scripts/           Train, evaluate, and MLflow server entrypoints
src/api/           FastAPI endpoints, schemas, registry model loading
src/data/          Validation, preparation, preprocessing, data splits
src/features/      Feature engineering
src/models/        Residual model and evaluation
src/mlflow_config.py Shared MLflow configuration
tests/             API and pipeline tests
```

Keep `.env`, `mlflow.db`, `mlartifacts/`, virtual environments, and local
datasets out of Git. `.env.example` documents the local environment variables;
never commit real credentials.
