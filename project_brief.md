# AirPulse — Project Brief

## Purpose

AirPulse is an academic machine-learning application that forecasts continuous
PM2.5 concentration one hour ahead for three Delhi air-quality stations. It is
an MLOps learning project, not an official air-quality warning service.

## Data and prediction target

- **Dataset:** XKDR Forum India Air Quality Database.
- **Period:** 2024 hourly observations.
- **Stations:** `site_113` (Shadipur), `site_115` (NSIT Dwarka), and
  `site_117` (ITO).
- **Target:** `target_pm25_next_hour`, in µg/m³; this is not AQI.
- **Horizon:** Predict at `t` using information available at `t`, for `t+1`.

The pipeline creates historical lag, rolling and calendar features. It retains
the current PM2.5 value (`pm25_at_origin`) as the persistence anchor. Data is
split chronologically by target time; the held-out test period is not used for
model selection. Preprocessing is fitted on training data only.

## Models and evaluation

The persistence baseline predicts `pm25_at_origin`. The machine-learning
candidate is a persistence-anchored residual HistGradientBoosting model:

```text
residual = actual_next_hour_pm25 - pm25_at_origin
prediction = pm25_at_origin + predicted_residual
```

It uses `loss="absolute_error"`, learning rate `0.05`, 300 iterations, 31
maximum leaf nodes, 20 minimum samples per leaf, L2 regularization `1.0`, no
early stopping, and random state `42`. Numeric features are unscaled and
unimputed; `station_id` is most-frequent imputed and one-hot encoded. The
preprocessor and estimator are saved together.

The reproduced matched validation comparison has 3,629 rows:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 10.7629 | 20.6201 | 0.8652 |
| Residual HGB | 10.2357 | 20.4997 | 0.8668 |

The saved historical test report contains 3,900 rows:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 20.0616 | 30.4415 | 0.9251 |
| Residual HGB | 21.4392 | 33.8036 | 0.9077 |

The current artifact improves all three overall validation metrics by a small
margin. The saved test report favors persistence, but its marker predates the
current model artifact; those results cannot be attributed to the current
artifact and are historical only. The test data has already been inspected and
must not be reused. A new untouched holdout period is required for a final
assessment of the current model. See
`docs/model_evaluation_and_current_selection.md` for details. The older
persistence-only report uses rows before 18-feature model-ready filtering, so
its counts and metrics are not a like-for-like comparison with these matched
model results.

## Application and MLOps components

- **API:** FastAPI (`main.py`, `src/api/`), with `/health`, `/predict`, and
  interactive API documentation at `/docs`.
- **Dashboard:** React, TypeScript and Vite (`frontend/`). It currently sends
  demonstration inputs; historical observations are not live-fed, and the
  historical chart is a placeholder.
- **Experiment tracking:** MLflow local store (`mlruns/`).
- **Automation and packaging:** GitHub Actions, Docker, and pytest.

## Reproducibility

From the repository root, with dependencies installed:

```powershell
python train.py
python -m scripts.evaluate
python -m src.models.verify_residual_hgb
python -m pytest
```

The test evaluator is guarded against repeat runs. Do not remove its marker or
rerun it. The canonical fitted pipeline and metadata are in `artifacts/models/`.

For data acquisition, set `XKDR_API_TOKEN` in a local `.env` file based on
`.env.example`; do not commit `.env`. More detailed dataset, quality,
preprocessing, feature, split, baseline and evaluation evidence is in `docs/`.
