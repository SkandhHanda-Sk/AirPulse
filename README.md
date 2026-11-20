# AirPulse

Station-wise next-hour PM2.5 forecasting for Delhi using 2024 observations from
the XKDR Forum India Air Quality Database:

| Station ID | Location |
|---|---|
| `site_113` | Shadipur |
| `site_115` | NSIT Dwarka |
| `site_117` | ITO |

The target is continuous PM2.5 concentration (`target_pm25_next_hour`), not AQI.
Input features describe information available at time `t`; the forecast is for
`t+1`. The train/validation/test split is chronological.

## Pipeline

```text
data/raw -> data/processed -> src/models -> artifacts/models
                                      -> FastAPI -> React dashboard
```

- **Baseline:** persistence, predicting `pm25_at_origin`.
- **ML candidate:** persistence-anchored residual HistGradientBoosting.
- **Residual target:** actual next-hour PM2.5 minus `pm25_at_origin`.
- **Final prediction:** `pm25_at_origin + predicted_residual`.
- Numeric features pass through unchanged. `station_id` is most-frequent
  imputed and one-hot encoded. Preprocessing is fit on training data only.
- Canonical model artifact: `artifacts/models/residual_hgb_pipeline.joblib`.
  The root `models/` folder is retained as a legacy compatibility location.
- MLflow tracks runs locally under `mlruns/`; that directory is not intended
  for Git commits.

For inference from the repository root, prepare a pandas DataFrame `X` with the
18 model-ready feature columns:

```python
from pathlib import Path

import joblib
from src.models.residual_hgb import EXPECTED_FEATURES

model = joblib.load(Path("artifacts") / "models" / "residual_hgb_pipeline.joblib")
predictions = model.predict(X.loc[:, EXPECTED_FEATURES])
```

## Results

Validation uses 3,629 matched model-ready rows. No rows were excluded from
that matched comparison.

| Split / Metric | Persistence | Residual HGB |
|---|---:|---:|
| Validation MAE | 10.7629 | 10.2357 |
| Validation RMSE | 20.6201 | 20.4997 |
| Validation R² | 0.8652 | 0.8668 |
| Previously recorded test MAE | 20.0616 | 21.4392 |
| Previously recorded test RMSE | 30.4415 | 33.8036 |
| Previously recorded test R² | 0.9251 | 0.9077 |

The current artifact was re-evaluated on validation data only; the residual
model improves all three overall validation metrics by a small margin. The
saved test report contains 3,900 matched rows and favors persistence. However,
its marker predates the current model artifact, so those results are historical
and cannot be attributed to the current artifact. The test data has already
been inspected; do not rerun evaluation or use it for tuning. A new untouched
holdout period is needed for a final assessment of the current model.

## Setup and run

Use Python 3.11 and Node.js. Install Python dependencies from the repository
root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train and evaluate the residual candidate on validation data:

```powershell
python train.py
python -m scripts.evaluate
python -m src.models.verify_residual_hgb
python -m pytest
```

Start the API from the repository root:

```powershell
python -m uvicorn main:app --reload
```

API endpoints are `/health` and `/predict`; interactive documentation is at
`/docs`.

Start the frontend in a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

The dashboard currently sends demonstration feature values to the working API.
It does not load recent station measurements, and its historical trend panel is
a placeholder; displayed predictions are not live forecasts.

To download the dataset, copy `.env.example` to `.env` and set
`XKDR_API_TOKEN`. Never commit `.env`. Existing local data and processed splits
are described in the reports under `docs/`.

When starting from a fresh dataset, run the data pipeline from the repository
root in this order (the downloader refuses to overwrite existing monthly files):

```powershell
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

## Model Quality Monitoring & Retraining (MLOps Assessment)

This repository includes a continuous model-quality monitoring and retraining pipeline (Tasks A–E):

```text
Incoming Batch -> [Task A] Schema & Hygiene Validation -> monitoring/data/new_batch.csv
              -> [Task B] Champion Evaluation (MAE, RMSE, R²) -> monitoring/reports/evaluation_report.json
              -> [Task C] Candidate Retraining + MLflow Tracking -> artifacts/models/candidate_model.joblib
              -> [Task D] Quality Gate (MAE ≤ Champion × 1.02) -> monitoring/reports/quality_gate_decisions.jsonl
              -> [Task E] Execution Evidence, Audit Trail & Documentation
```

### Running the Monitoring Workflow

Run end-to-end via command line:

```powershell
python monitoring/run_monitoring_pipeline.py
```

Or open and run the interactive Jupyter notebook:

```powershell
jupyter notebook monitoring/monitoring_retraining.ipynb
```

Run automated monitoring unit tests:

```powershell
pytest tests/test_monitoring.py
```

### Quality Gate Decision Rule & Metric Justification

- **Primary Metric:** Mean Absolute Error (MAE, in µg/m³).
- **Rule:** `Candidate MAE <= Champion MAE * 1.02` (2% tolerance).
- **Justification:**
  1. Internal consistency: The model is trained with L1 absolute loss (`least_absolute_deviation` / `absolute_error`).
  2. Domain interpretability: Error is directly in µg/m³, mapping to CPCB / WHO health advisory bands.
  3. Robustness: Avoids quadratic inflation from episodic PM2.5 spikes common in Delhi.
  4. Non-destructive safety guardrail: The champion model is never automatically overwritten without human sign-off.

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
