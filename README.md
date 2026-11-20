# AirPulse

AirPulse forecasts continuous PM2.5 concentration one hour ahead for three
Delhi monitoring stations using 2024 observations from the XKDR Forum India Air
Quality Database. The target is PM2.5 concentration, not AQI. This is an
academic forecasting application, not an official air-quality warning service.

| Station ID | Location |
|---|---|
| `site_113` | Shadipur |
| `site_115` | NSIT Dwarka |
| `site_117` | ITO |

## Model and evaluation

The model is a persistence-anchored residual HistGradientBoosting regressor:

```text
residual   = actual_next_hour_pm25 - pm25_at_origin
prediction = pm25_at_origin + predicted_residual
```

The chronological training, validation, and test splits and the existing
18-feature preprocessing pipeline are preserved. Numeric features pass through
unchanged; `station_id` is imputed and one-hot encoded. Preprocessing is fit on
training data only.

The current validation comparison uses 3,629 matched rows:

| Metric | Persistence | Residual HGB |
|---|---:|---:|
| MAE | 10.7629 | 10.2357 |
| RMSE | 20.6201 | 20.4997 |
| R² | 0.8652 | 0.8668 |

The existing saved test report is historical and predates the current model
artifact. Test data has already been inspected; do not use it for tuning or
repeat the guarded test evaluation. Use a new untouched holdout period for a
future final assessment.

## MLflow tracking and model registry

The project uses MLflow **2.14.1** with a SQLite backend store and a local
artifact directory. `MLFLOW_TRACKING_URI` is the configuration source; the
default is `http://127.0.0.1:7001`. The experiment is
`AirPulse_PM25_Forecasting`. The existing registered model name is
`airpulse-pm25-forecast`.

Each `python train.py` run logs parameters, training and validation metrics,
run metadata, and the serialized model. It registers a new model version.
Promotion uses the existing validation methodology: the candidate passes only
when its validation MAE is lower than the persistence baseline MAE on the same
validation rows. There is no fixed absolute cutoff. A failed candidate is
registered but does not replace the current `champion` alias. MLflow errors
are surfaced rather than ignored.

The API loads only the registered champion:

```text
models:/airpulse-pm25-forecast@champion
```

## Run the full local application (Windows CMD)

Use four CMD windows. Keep the MLflow server, API, and frontend windows open.
Run training after MLflow is ready and before starting the API.
In each window, replace `C:\path\to\AirPulse` with the folder where you
extracted the project.

### 1. MLflow server

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 7001 --default-artifact-root ./mlartifacts
```

Open the tracking UI at **http://127.0.0.1:7001**.

### 2. Train and evaluate

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python train.py
python -m scripts.evaluate
```

The training run evaluates the candidate, registers its version, and assigns
`champion` only after a passing gate. The separate evaluation command records
another validation run.

### 3. FastAPI backend (use this for the current checkout)

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%"
call .venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://127.0.0.1:7001
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check: **http://127.0.0.1:8000/health**

Interactive API docs: **http://127.0.0.1:8000/docs**

### 4. React dashboard

```cmd
set "AIRPULSE_ROOT=C:\path\to\AirPulse"
cd /d "%AIRPULSE_ROOT%\frontend"
npm run dev -- --host 127.0.0.1
```

Open the Vite URL, normally **http://127.0.0.1:5173**. Click **Get next-hour
prediction** to request a forecast. The dashboard currently sends demonstration
feature values; it does not fetch live station measurements, and the historical
panel is a placeholder.

### Optional Docker backend (no image build)

MLflow must be running on the host first. To start the existing container:

```cmd
docker start airpulse-api
docker logs -f airpulse-api
```

The container uses `host.docker.internal:7001` to reach host MLflow and maps the
API to port 8000. Use either this container **or** the local Uvicorn command;
do not run both at once. An existing image does not include later source
changes. No Docker Compose file is currently provided.

## Verify the registry in MLflow UI

1. Open **http://127.0.0.1:7001** and select `AirPulse_PM25_Forecasting`.
2. Open a `residual_hgb_training` run to inspect its parameters, metrics, and
   `model` and `metadata` artifacts.
3. Open **Models** → `airpulse-pm25-forecast`.
4. Confirm the new version is `READY` and the `champion` alias points to the
   latest candidate that passed validation.

## Initial setup

Use Python 3.11 and Node.js. Install dependencies once from the repository
root if they are not already installed:

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
cd frontend
npm ci
```

For fresh data acquisition, put `XKDR_API_TOKEN` in a local `.env` based on
`.env.example`; do not commit `.env`. Run the existing data pipeline from the
repository root only when preparing a fresh dataset:

```cmd
python -m src.data.download_xkdr
python -m src.data.preprocess
python -m src.features.build_features
python -m src.data.split_dataset
python -m src.data.prepare_model_data
```

Then train and evaluate using the commands above. Keep the existing test split
out of model selection; a new untouched holdout period is required for a valid
final evaluation of the current artifact.

The test-evaluation command is guarded against repeat runs. Do not remove its
marker or rerun it; the saved test report is historical and does not verify the
current artifact.

## Repository map

- `src/data/`, `src/features/`, `src/models/`: data, feature and model pipeline.
- `src/api/`, `backend/`: prediction API and compatibility entrypoint.
- `scripts/`: MLflow training and validation evaluation commands.
- `frontend/src/`: dashboard pages, components, styles and API client.
- `artifacts/models/`: canonical serialized model and metadata.
- `archive/experiments/`: retained, inactive validation-only experiment runner.
- `docs/`: dataset, preprocessing, feature, split, baseline and evaluation
  records.
- `tests/`: API and pipeline tests.

## Continuous integration

GitHub Actions runs the automated test, frontend build, and Docker packaging
checks for pushes and pull requests.
